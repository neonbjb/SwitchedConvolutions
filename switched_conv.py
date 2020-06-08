import functools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


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
            "...ij,...j->...i", [conv_outputs.to(dtype=torch.float32), conv_attention]
        )

        # Remember to shift the filters back into the expected slot.
        if output_attention_weights:
            return attention_result.permute(0, 3, 1, 2), conv_attention
        else:
            return attention_result.permute(0, 3, 1, 2)


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
