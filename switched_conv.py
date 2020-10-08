import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .switched_conv_util import initialize_weights

"""
AttentionNorm is meant to be applied against the Softmax output of an attention function across a large set of
attention computations. It is meant to promote an equal distribution of attention weights by decreasing the magnitude
of attention weights that are over-used and increasing the magnitude of under-used weights.

The return value has the exact same format as a normal Softmax output and can be used directly into the input of an
attention equation.

Since the whole point of convolutional attention is to enable training extra-wide networks to operate on a large number
of image categories, it makes almost no sense to perform this type of norm against a single mini-batch of images: some
of the attentions will not be used in such a small context - and that's good! This is solved by accumulating. Every 
forward pass computes a norm across the current minibatch. That norm is added into a rotating buffer of size 
<accumulator_size>. The actual normalization occurs across the entire rotating buffer.

You should set accumulator size according to two factors:
- Your batch size. Smaller batch size should mean greater accumulator size.
- Your image diversity. More diverse images have less need for the accumulator.
- How wide your attention/switching group size is. More groups mean you're going to want more accumulation.

Note that this norm does nothing for the first <accumulator_size> iterations.
"""
class AttentionNorm(nn.Module):
    def __init__(self, group_size, accumulator_size=128):
        super(AttentionNorm, self).__init__()
        self.accumulator_desired_size = accumulator_size
        self.group_size = group_size
        # These are all tensors so that they get saved with the graph.
        self.accumulator = nn.Parameter(torch.zeros(accumulator_size, group_size), requires_grad=False)
        self.accumulator.DO_NOT_TRAIN = True
        self.accumulator_index = nn.Parameter(torch.zeros(1, dtype=torch.long, device='cpu'), requires_grad=False)
        self.accumulator_filled = nn.Parameter(torch.zeros(1, dtype=torch.long, device='cpu'), requires_grad=False)

    # Returns tensor of shape (group,) with a normalized mean across the accumulator in the range [0,1]. The intent
    # is to divide your inputs by this value.
    def compute_buffer_norm(self):
        if self.accumulator_filled > 0:
            return torch.mean(self.accumulator, dim=0)
        else:
            return torch.ones(self.group_size, device=self.accumulator.device)

    def add_norm_to_buffer(self, x):
        flat = x.sum(dim=[0, 1, 2], keepdim=True)
        norm = flat / torch.mean(flat)

        # This often gets reset in GAN mode. We *never* want gradient accumulation in this parameter.
        self.accumulator[self.accumulator_index] = norm.detach().clone()
        self.accumulator_index += 1
        if self.accumulator_index >= self.accumulator_desired_size:
            self.accumulator_index *= 0
            if self.accumulator_filled <= 0:
                self.accumulator_filled += 1

    # Input into forward is an attention tensor of shape (batch,width,height,groups)
    def forward(self, x: torch.Tensor, update_attention_norm=True):
        assert len(x.shape) == 4
        # Push the accumulator to the right device on the first iteration.
        if self.accumulator.device != x.device:
            self.accumulator = self.accumulator.to(x.device)

        # In eval, don't change the norm buffer.
        if self.training and update_attention_norm:
            self.add_norm_to_buffer(x)
        norm = self.compute_buffer_norm()
        x = x / norm

        # Need to re-normalize x so that the groups dimension sum to 1, just like when it was fed in.
        groups_sum = x.sum(dim=3, keepdim=True)
        return x / groups_sum

    def reduce_params(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.accumulator)
            self.accumulator /= dist.get_world_size()

    def load_state_dict(self, state_dict, strict=True):

        # The parameters in self.trunk used to be in this class. To support loading legacy saves, restore them.
        t_state = self.trunk.state_dict()
        for k in t_state.keys():
            if k in state_dict.keys():
                state_dict["trunk.%s" % (k,)] = state_dict.pop(k)
        super(RRDBNet, self).load_state_dict(state_dict, strict)


class BareConvSwitch(nn.Module):
    """
    Initializes the ConvSwitch.
      initial_temperature: The initial softmax temperature of the attention mechanism. For training from scratch, this
                           should be set to a high number, for example 30.
      attention_norm:      If specified, the AttentionNorm layer applied immediately after Softmax.
    """
    def __init__(
        self,
        initial_temperature=1,
        attention_norm=None
    ):
        super(BareConvSwitch, self).__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.temperature = initial_temperature
        self.attention_norm = attention_norm

        initialize_weights(self)

    def reduce_norm_params(self):
        self.attention_norm.reduce_params()
        
    def set_attention_temperature(self, temp):
        self.temperature = temp

    # SwitchedConv.forward takes these arguments;
    # conv_group:      List of inputs (len=n) to the switch, each with shape (b,f,w,h)
    # conv_attention:  Attention computation as an output from a conv layer, of shape (b,n,w,h). Before softmax
    # output_attention_weights: If True, post-softmax attention weights are returned.
    def forward(self, conv_group, conv_attention, output_attention_weights=False, update_attention_norm=True):
        # Stack up the conv_group input first and permute it to (batch, width, height, filter, groups)
        conv_outputs = torch.stack(conv_group, dim=0).permute(1, 3, 4, 2, 0)

        conv_attention = conv_attention.permute(0, 2, 3, 1)
        conv_attention = self.softmax(conv_attention / self.temperature)
        if self.attention_norm:
            conv_attention = self.attention_norm(conv_attention, update_attention_norm)

        # conv_outputs shape:   (batch, width, height, filters, groups)
        # conv_attention shape: (batch, width, height, groups)
        # We want to format them so that we can matmul them together to produce:
        # desired shape:        (batch, width, height, filters)
        # Note: conv_attention will generally be cast to float32 regardless of the input type, so cast conv_outputs to
        #       float32 as well to match it.
        if self.training:
            # Doing it all in one op is substantially faster - better for training.
            attention_result = torch.einsum(
                "...ij,...j->...i", [conv_outputs.float(), conv_attention]
            )
        else:
            # eval_mode substantially reduces the GPU memory required to compute the attention result by performing the
            # attention multiplications one at a time. This is probably necessary for large images and attention breadths.
            attention_result = conv_outputs[:, :, :, :, 0] * conv_attention[:, :, :, 0].unsqueeze(dim=-1)
            for i in range(1, conv_attention.shape[-1]):
                attention_result += conv_outputs[:, :, :, :, i] * conv_attention[:, :, :, i].unsqueeze(dim=-1)

        # Remember to shift the filters back into the expected slot.
        if output_attention_weights:
            return attention_result.permute(0, 3, 1, 2), conv_attention
        else:
            return attention_result.permute(0, 3, 1, 2)


class ConvSwitch(nn.Module):
    """
    Initializes the ConvSwitch.
      nf_attention_basis: Number of filters provided to the attention_basis input of forward(). Must be divisible by two
                          and nf_attention_basis/2 >= num_convs.
      num_convs:          Number of elements that will appear in the conv_group() input. The attention mechanism will
                          select across this number.
      att_kernel_size:    The size of the attention mechanisms convolutional kernels.
      att_stride:         The stride of the attention mechanisms conv blocks.
      att_pads:           The padding of the attention mechanisms conv blocks.
      att_interpolate_scale_factor:
                          The scale factor applied to the attention mechanism's outputs.
      *** NOTE ***: Between stride, pads, and interpolation_scale_factor, the output of the attention mechanism MUST
                    have the same width/height as the conv_group inputs.
      initial_temperature: The initial softmax temperature of the attention mechanism. For training from scratch, this
                           should be set to a high number, for example 30.
    """

    def __init__(
        self,
        nf_attention_basis,
        num_convs=8,
        att_kernel_size=5,
        att_stride=1,
        att_pads=2,
        att_interpolate_scale_factor=1,
        initial_temperature=1,
    ):
        super(ConvSwitch, self).__init__()

        # Requirements: input filter count is even, and there are more filters than there are sequences to attend to.
        assert nf_attention_basis % 2 == 0
        assert nf_attention_basis / 2 >= num_convs

        self.num_convs = num_convs
        self.interpolate_scale_factor = att_interpolate_scale_factor
        self.attention_conv1 = nn.Conv2d(
            nf_attention_basis,
            int(nf_attention_basis / 2),
            att_kernel_size,
            att_stride,
            att_pads,
            bias=True,
        )
        self.att_bn1 = nn.BatchNorm2d(int(nf_attention_basis / 2))
        self.attention_conv2 = nn.Conv2d(
            int(nf_attention_basis / 2),
            num_convs,
            att_kernel_size,
            1,
            att_pads,
            bias=True,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.temperature = initial_temperature

        initialize_weights(self)

    def set_attention_temperature(self, temp):
        self.temperature = temp

    # SwitchedConv.forward takes these arguments;
    # attention_basis: The tensor to compute attention vectors from. Generally this should be the original inputs of
    #                  your conv group.
    # conv_group:      A list of output tensors from the convolutional groups that the attention mechanism is selecting
    #                  from. Each tensor in this list is expected to have the shape (batch, filters, width, height)
    def forward(self, attention_basis, conv_group, output_attention_weights=False):
        assert self.num_convs == len(conv_group)

        # Stack up the conv_group input first and permute it to (batch, width, height, filter, groups)
        conv_outputs = torch.stack(conv_group, dim=0).permute(1, 3, 4, 2, 0)

        # Now calculate the attention across those convs.
        conv_attention = self.lrelu(self.att_bn1(self.attention_conv1(attention_basis)))
        conv_attention = self.attention_conv2(conv_attention).permute(0, 2, 3, 1)
        conv_attention = self.softmax(conv_attention / self.temperature)

        # Interpolate to (hopefully) match the input conv_group.
        if self.interpolate_scale_factor != 1:
            conv_attention = F.interpolate(
                conv_attention,
                scale_factor=self.interpolate_scale_factor,
                mode="nearest",
            )

        # conv_outputs shape:   (batch, width, height, filters, groups)
        # conv_attention shape: (batch, width, height, groups)
        # We want to format them so that we can matmul them together to produce:
        # desired shape:        (batch, width, height, filters)
        # Note: conv_attention will generally be cast to float32 regardless of the input type, so cast conv_outputs to
        #       float32 as well to match it.
        attention_result = torch.einsum(
            "...ij,...j->...i", [conv_outputs.float(), conv_attention]
        )

        # Remember to shift the filters back into the expected slot.
        if output_attention_weights:
            return attention_result.permute(0, 3, 1, 2), conv_attention
        else:
            return attention_result.permute(0, 3, 1, 2)


'''
This is a debug function for the attention mechanism used by ConvSwitch.

Pulls the top k values from each attention vector across the image (b,w,h) and computes the mean. This value is termed
the "specificity" and represents how much attention is paid to the top-k filters. It ranges between [0,1].

Also returns a flat list of indices that represent the top-k attention values. These can be fed into a histogram to
see how the model is utilizing the computational blocks underlying the switch.
'''
def compute_attention_specificity(att_weights, topk=3):
    att = att_weights.detach()
    vals, indices = torch.topk(att, topk, dim=-1)
    avg = torch.sum(vals, dim=-1)
    avg = avg.flatten().mean()
    return avg.item(), indices.flatten().detach()


"""
 Implements convolutional switching across an abstract block. The block should be provided as follows:
 functools.partial(<block>, <constructor parameters>)
 See the next class for a sample.
"""
class SwitchedAbstractBlock(nn.Module):
    def __init__(
        self,
        partial_block_constructor,
        nf_attention_basis,
        num_blocks=8,
        att_kernel_size=5,
        att_stride=1,
        att_pads=2,
        att_interpolate_scale_factor=1,
        initial_temperature=1,
    ):
        super(SwitchedAbstractBlock, self).__init__()
        self.switcher = ConvSwitch(
            nf_attention_basis,
            num_blocks,
            att_kernel_size,
            att_stride,
            att_pads,
            att_interpolate_scale_factor,
            initial_temperature,
        )
        self.block_list = nn.ModuleList(
            [partial_block_constructor() for _ in range(num_blocks)]
        )

    def set_attention_temperature(self, temp):
        self.switcher.set_attention_temperature(temp)

    def forward(self, x, output_attention_weights=False):
        # Build up the individual conv components first.
        block_outputs = []
        for block in self.block_list:
            block_outputs.append(block.forward(x))
        return self.switcher.forward(x, block_outputs, output_attention_weights)


# Implements a basic Conv2d block which is backed by a switching mechanism.
class SwitchedConv2d(SwitchedAbstractBlock):
    def __init__(
        self,
        nf_in_per_conv,
        nf_out_per_conv,
        kernel_size,
        stride=1,
        pads=0,
        num_convs=8,
        att_kernel_size=5,
        att_padding=2,
        initial_temperature=1,
    ):
        partial_block = functools.partial(
            nn.Conv2d, nf_in_per_conv, nf_out_per_conv, kernel_size, stride, pads
        )
        super(SwitchedConv2d, self).__init__(
            partial_block,
            nf_in_per_conv,
            num_convs,
            att_kernel_size,
            stride,
            att_padding,
            initial_temperature=initial_temperature,
        )

"""
 Implements multi-headed convolutional switching across an abstract block. The block should be provided as follows:
 functools.partial(<block>, <constructor parameters>)
 See the next class for a sample.
"""
class MultiHeadSwitchedAbstractBlock(nn.Module):

    '''
    partial_block_constructor: functools.partial() of a block that performs processing on an image of shape (b,nf_attention_basis,w,h)
    nf_attention_basis: the filter count that will be fed into this block.
    num_blocks: the number of parallel blocks to compute
    num_heads: the number of switch heads to bring to bear against the blocks.
    att_kernel_size: the size of the convolutional kernel used to compute the switching probabilities.
    att_stride: this is the stride for the switching kernel. It should match the aggregate stride of blocks created by partial_block_constructor
    att_pads: likewise, stride for the switching convolution.
    att_interpolate_scale_factor: Can be used to upsample the output of the switching conv.
    initial_temperature: Controls the initial setting of the switching softmax.
    include_skip_head: If True, the last head will not use the block, but will instead feed-forward the input. Only works if the input and output dimensions of the processing block are equal.
    multi_head_input: When true, expect input of shape (b,heads,f,w,h), else (b,f,w,h)
    concat_heads_into_filter: When true, return from this is shape (b,f*heads,w,h). When false, it is (b,heads,f,w,h)
    '''
    def __init__(
        self,
        partial_block_constructor,
        nf_attention_basis,
        num_blocks=8,
        num_heads=2,
        att_kernel_size=5,
        att_stride=1,
        att_pads=2,
        att_interpolate_scale_factor=1,
        initial_temperature=1,
        include_skip_head=False,
        multi_head_input=False,
        concat_heads_into_filters=True
    ):
        super(MultiHeadSwitchedAbstractBlock, self).__init__()
        self.include_skip_head = include_skip_head
        if multi_head_input:
            processing_blocks = num_blocks - num_heads if self.include_skip_head else num_blocks
        else:
            processing_blocks = num_blocks - 1 if self.include_skip_head else num_blocks
        self.block_list = nn.ModuleList(
            [partial_block_constructor() for _ in range(processing_blocks)]
        )
        self.switches = nn.ModuleList([ConvSwitch(nf_attention_basis, num_blocks, att_kernel_size, att_stride, att_pads, att_interpolate_scale_factor, initial_temperature) for i in range(num_heads)])
        self.concat_heads_into_filters = concat_heads_into_filters
        self.multi_head_input = multi_head_input
        self.num_heads = num_heads
        if self.multi_head_input:
            self.mhead_squash = nn.Conv3d(nf_attention_basis, nf_attention_basis, (num_heads, 1, 1), (num_heads, 1, 1))
            initialize_weights(self.mhead_squash)


    def forward(self, x, output_attention_weights=False):
        # Build up the individual conv components first.
        block_outputs = []
        for block in self.block_list:
            block_outputs.append(block.forward(x))
        if self.include_skip_head:
            if self.multi_head_input:
                for h in range(self.num_heads):
                    block_outputs.append(x[:, h, :])
            else:
                block_outputs.append(x)

        # Squash input heads before feeding into attention switches.
        if self.multi_head_input:
            x = x.permute(0, 2, 1, 3, 4)
            x = self.mhead_squash(x)
            x = torch.squeeze(x, dim=2)

        outs = []
        atts = []
        for switch in self.switches:
            out, att = switch.forward(x, block_outputs, output_attention_weights)
            outs.append(out)
            atts.append(att)

        if self.concat_heads_into_filters:
            out = torch.cat(outs, 1)
        else:
            out = torch.stack(outs, 1)
        # The output will be the heads concatenated across the filter dimension. Attention outputs will be stacked into
        # a new dimension.
        if output_attention_weights:
            return out, torch.stack(atts, 1)
        else:
            return out

    def set_attention_temperature(self, temp):
        for switch in self.switches:
            switch.set_attention_temperature(temp)
