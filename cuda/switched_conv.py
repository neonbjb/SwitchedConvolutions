import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lambda_networks import LambdaLayer
from torch.nn import init, Conv2d, ZeroPad2d


def SwitchedConvRoutingNormal(input, selector, weight, bias, stride=1):
    convs = []
    b, s, h, w = selector.shape
    for sel in range(s):
        convs.append(F.conv2d(input, weight[:, :, sel, :, :], bias, stride=stride, padding=weight.shape[-1] // 2))
    output = torch.stack(convs, dim=1) * selector.unsqueeze(dim=2)
    return output.sum(dim=1)


class SwitchedConvHardRoutingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, selector, weight, bias, stride=1):
        # Pre-pad the input.
        input = ZeroPad2d(weight.shape[-1]//2)(input)

        # Build hard attention mask from selector input
        b, s, h, w = selector.shape

        mask = selector.argmax(dim=1).int()
        import switched_conv_cuda_naive
        output = switched_conv_cuda_naive.forward(input, mask, weight, bias, stride)

        ctx.stride = stride
        ctx.breadth = s
        ctx.save_for_backward(*[input, output.detach().clone(), mask, weight, bias])
        return output

    @staticmethod
    def backward(ctx, gradIn):
        #import pydevd   # Uncomment to allow debugging inside this function.
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)
        input, output, mask, weight, bias = ctx.saved_tensors
        gradIn = gradIn

        # Selector grad is simply the element-wise product of grad with the output of the layer, summed across the channel dimension
        # and repeated along the breadth of the switch. (Think of the forward operation using the selector as a simple matrix of 1s
        # and zeros that is multiplied by the output.)
        grad_sel = (gradIn * output).sum(dim=1, keepdim=True).repeat(1,ctx.breadth,1,1)

        import switched_conv_cuda_naive
        grad, grad_w, grad_b = switched_conv_cuda_naive.backward(input, gradIn.contiguous(), mask, weight, bias, ctx.stride)

        # Remove input padding from grad
        padding = weight.shape[-1] // 2
        if padding > 0:
            grad = grad[:,:,padding:-padding,padding:-padding]
        return grad, grad_sel, grad_w, grad_b, None


"""
Creates a hard routing attention tensor which properly routes gradients in the backwards pass. 

Accomplished by finding the argmax of the elements in the input (across dim=1) and setting those elements to 1. All 
others are set to 0. 

In the backwards pass, the gradient is fed directly to the elements set to 1 only. The gradients for those elements are 
scaled by the original input value (as if the stepwise function setting the inputs to 1 didn't happen).
"""
class RouteTop1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        mask = torch.nn.functional.one_hot(input.argmax(dim=1), num_classes=input.shape[1]).permute(0,3,1,2)
        out = torch.ones_like(input)
        out[mask != 1] = 0
        ctx.save_for_backward(mask, input.clone())
        return out

    @staticmethod
    def backward(ctx, grad):
        # Enable breakpoints in this function:  (Comment out if not debugging)
        #import pydevd
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)

        mask, input = ctx.saved_tensors
        input[mask != 1] = 1
        grad_input = grad.clone()
        grad_input[mask != 1] = 0
        grad_input_n = grad_input / input  # Above, we made everything either a zero or a one. Unscale the ones by dividing by the unmasked inputs.
        return grad_input_n


"""
SwitchNorm is meant to be applied against the Softmax output of a switching function across a large set of
switch computations. It is meant to promote an equal distribution of switch weights by decreasing the magnitude
of switch weights that are over-used and increasing the magnitude of under-used weights.

The return value has the exact same format as a normal Softmax output and can be used directly into the input of an
switch equation.

Since the whole point of convolutional switch is to enable training extra-wide networks to operate on a large number
of image categories, it makes almost no sense to perform this type of norm against a single mini-batch of images: some
of the switches will not be used in such a small context - and that's good! This is solved by accumulating. Every 
forward pass computes a norm across the current minibatch. That norm is added into a rotating buffer of size 
<accumulator_size>. The actual normalization occurs across the entire rotating buffer.

You should set accumulator size according to two factors:
- Your batch size. Smaller batch size should mean greater accumulator size.
- Your image diversity. More diverse images have less need for the accumulator.
- How wide your switch/switching group size is. More groups mean you're going to want more accumulation.

Note: This norm makes the (potentially flawed) assumption that each forward() pass has unique data. For maximum 
      effectiveness, avoid performing regular, repeated forward passes with the same data - or make alterations to work 
      around it.
Note: This norm does nothing for the first <accumulator_size> iterations.
"""
class SwitchNorm(nn.Module):
    def __init__(self, group_size, accumulator_size=128):
        super().__init__()
        self.accumulator_desired_size = accumulator_size
        self.group_size = group_size
        self.register_buffer("accumulator_index", torch.zeros(1, dtype=torch.long, device='cpu'))
        self.register_buffer("accumulator_filled", torch.zeros(1, dtype=torch.long, device='cpu'))
        self.register_buffer("accumulator", torch.zeros(accumulator_size, group_size))

    def add_norm_to_buffer(self, x):
        flat = x.sum(dim=[0, 2, 3])
        norm = flat / torch.mean(flat)

        self.accumulator[self.accumulator_index] = norm.detach().clone()
        self.accumulator_index += 1
        if self.accumulator_index >= self.accumulator_desired_size:
            self.accumulator_index *= 0
            if self.accumulator_filled <= 0:
                self.accumulator_filled += 1

    # Input into forward is a switching tensor of shape (batch,groups,width,height)
    def forward(self, x: torch.Tensor, update_attention_norm=True):
        assert len(x.shape) == 4

        # Push the accumulator to the right device on the first iteration.
        if self.accumulator.device != x.device:
            self.accumulator = self.accumulator.to(x.device)

        # In eval, don't change the norm buffer.
        if self.training and update_attention_norm:
            self.add_norm_to_buffer(x)

        # Reduce across all distributed entities, if needed
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.accumulator, op=dist.ReduceOp.SUM)
            self.accumulator /= dist.get_world_size()

        # Compute the norm factor.
        if self.accumulator_filled > 0:
            norm = torch.mean(self.accumulator, dim=0)
        else:
            norm = torch.ones(self.group_size, device=self.accumulator.device)
        x = x / norm.view(1,-1,1,1)

        # Need to re-normalize x so that the groups dimension sum to 1, just like when it was fed in.
        return x / x.sum(dim=1, keepdim=True)


class HardRoutingGate(nn.Module):
    def __init__(self, breadth, hard_en=True):
        super().__init__()
        self.norm = SwitchNorm(breadth, accumulator_size=256)
        self.hard_en = hard_en

    def forward(self, x):
        soft = self.norm(nn.functional.softmax(x, dim=1))
        if self.hard_en:
            return RouteTop1.apply(soft)
        return soft


class SwitchedConvHardRouting(nn.Module):
    def __init__(self,
                 name,
                 in_c,
                 out_c,
                 kernel_sz,
                 breadth,
                 stride=1,
                 bias=True,
                 dropout_rate=0.0,
                 include_coupler: bool = False,  # A 'coupler' is a latent converter which can transforms the input into a switch selector. For large networks using many SwitchedConvs, it is recommended to provide an external coupler.
                 coupler_mode: str = 'standard',
                 coupler_dim_in: int = 0,
                 hard_en=True):  # A test switch that, when used in 'emulation mode' (where all convs are calculated using torch functions) computes soft-attention instead of hard-attention.
        super().__init__()
        self.name = name
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_sz
        self.stride = stride
        self.has_bias = bias
        self.breadth = breadth
        self.dropout_rate = dropout_rate
        self.step_count = 0

        if coupler_dim_in == 0:
            coupler_dim_in = in_c
        if include_coupler:
            if coupler_mode == 'standard':
                self.coupler = Conv2d(coupler_dim_in, breadth, kernel_size=1, stride=self.stride)
            elif coupler_mode == 'lambda':
                self.coupler = nn.Sequential(nn.GroupNorm(num_groups=2, num_channels=coupler_dim_in),
                                             LambdaLayer(dim=coupler_dim_in, dim_out=breadth, r=23, dim_k=16, heads=2, dim_u=1),
                                             nn.ReLU(),
                                             Conv2d(breadth, breadth, 1, stride=self.stride))
            elif coupler_mode == 'lambda2':
                self.coupler = nn.Sequential(LambdaLayer(dim=coupler_dim_in, dim_out=coupler_dim_in, r=23, dim_k=16, heads=2, dim_u=1),
                                             nn.GroupNorm(num_groups=2, num_channels=coupler_dim_in),
                                             nn.ReLU(),
                                             LambdaLayer(dim=coupler_dim_in, dim_out=breadth, r=23, dim_k=16, heads=2, dim_u=1),
                                             nn.GroupNorm(num_groups=1, num_channels=breadth),
                                             nn.ReLU(),
                                             Conv2d(breadth, breadth, 1, stride=self.stride))
        else:
            self.coupler = None
        self.gate = HardRoutingGate(breadth, hard_en=False)
        self.hard_en = hard_en

        self.weight = nn.Parameter(torch.empty(out_c, in_c, breadth, kernel_sz, kernel_sz))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_c))
        else:
            self.bias = torch.zeros(out_c)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[:,:,0,:,:])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def load_weights_from_conv(self, cnv):
        sd = cnv.state_dict()
        sd['weight'] = sd['weight'].unsqueeze(2).repeat(1,1,self.breadth,1,1)
        self.load_state_dict(sd)

    def forward(self, input, selector=None):
        if self.bias.device != input.device:
            self.bias = self.bias.to(input.device)  # Because this bias can be a tensor that is not moved with the rest of the module.

        # If a coupler was specified, run that to convert selector into a softmax distribution.
        if self.coupler:
            if selector is None:  # A coupler can convert from any input to a selector, so 'None' is allowed.
                selector = input
            selector = self.coupler(selector)
        assert selector is not None

        # Apply dropout at the batch level per kernel.
        if self.training and self.dropout_rate > 0:
            b, c, h, w = selector.shape
            drop = torch.rand((b, c, 1, 1), device=input.device) > self.dropout_rate
            # Ensure that there is always at least one switch left un-dropped out
            fix_blank = (drop.sum(dim=1, keepdim=True) == 0).repeat(1, c, 1, 1)
            drop = drop.logical_or(fix_blank)
            selector = drop * selector

        selector = self.gate(selector)

        # Debugging variables
        if self.step_count % 200 == 0:
            os.makedirs('work_dirs/sw_debug', exist_ok=True)
            self.save_attention_to_image_rgb(os.path.join('work_dirs/sw_debug', "%s_selector_%i.png" % (self.name, self.step_count)), selector.detach().clone(), self.breadth)
        self.step_count += 1

        if False:
            # This is a custom CUDA implementation which should be faster and less memory intensive (once completed).
            return SwitchedConvHardRoutingFunction.apply(input, selector, self.weight, self.bias, self.stride)
        else:
            # This composes the switching functionality using raw Torch, which basically consists of computing each of <breadth> convs separately and combining them.
            return SwitchedConvRoutingNormal(input, selector, self.weight, self.bias, self.stride)

    def save_attention_to_image_rgb(self, output_file, attention_out, attention_size, cmap_discrete_name='viridis'):
        from matplotlib import cm
        magnitude, indices = torch.topk(attention_out, 3, dim=1)
        indices = indices.cpu()
        colormap = cm.get_cmap(cmap_discrete_name, attention_size)
        img = torch.tensor(colormap(indices[:, 0, :, :].detach().numpy()))  # TODO: use other k's
        img = img.permute((0, 3, 1, 2))
        torchvision.utils.save_image(img, output_file)


# Given a state_dict and the module that that sd belongs to, strips out the specified Conv2d modules and replaces them
# with equivalent switched_conv modules.
def convert_net_to_switched_conv(module, switch_breadth, allow_list, dropout_rate=0.4, coupler_mode='lambda'):
    full_paths = [n.split('.') for n in allow_list]
    for modpath in full_paths:
        mod = module
        for sub in modpath[:-1]:
            mod = getattr(mod, sub)
        old_conv = getattr(mod, modpath[-1])
        new_conv = SwitchedConvHardRouting('.'.join(modpath), old_conv.in_channels, old_conv.out_channels, old_conv.kernel_size[0], switch_breadth, old_conv.stride[0], old_conv.bias is not None,
                                           include_coupler=True, dropout_rate=dropout_rate, coupler_mode=coupler_mode)
        new_conv = new_conv.to(old_conv.weight.device)
        assert old_conv.dilation == 1 or old_conv.dilation == (1,1) or old_conv.dilation is None
        if isinstance(mod, nn.Sequential) or isinstance(mod, nn.ModuleList):
            # If we use the standard logic (in the else case) here, it reorders the sequential.
            # Instead, extract the OrderedDict from the current sequential and edit that.
            mod._modules[modpath[-1]] = new_conv
        else:
            delattr(mod, modpath[-1])
            mod.add_module(modpath[-1], new_conv)

def convert_state_dict_to_switched_conv(sd, switch_breadth, allow_list):
    converted = 0
    for cname in allow_list:
        for sn in sd.keys():
            if cname in sn and sn.endswith('weight'):
                # If you are getting an error here - it's likely you are trying to convert the weights for a model that has already been converted! If using SwitchedConvConversionWrapper, this can also mean that the state_dict you provided is not compatible with the converted model.
                sd[sn] = sd[sn].unsqueeze(2).repeat(1,1,switch_breadth,1,1)
                converted += 1
    print(f"Converted {converted} parameters.")
    return sd


class SwitchedConvConversionWrapper:
    def __init__(self, wrap_module, breadth, allow_list, coupler_mode='lambda', dropout_rate=0.4):
        self.wrapped_module = wrap_module
        self.breadth = breadth
        self.coupler_mode = coupler_mode
        self.allow_list = allow_list
        convert_net_to_switched_conv(self.wrapped_module, switch_breadth=breadth, allow_list=allow_list, coupler_mode=coupler_mode, dropout_rate=dropout_rate)

    def load_state_dict(self, state_dict, suppress_autoconvert_weights=False, strict = True):
        # We need to handle two cases here:
        # 1) The provided weights are for the unconverted model. We detect this by catching an exception from the torch
        #    implementation.
        # 2) The provided weights are for a converted model. We assume this at first.
        try:
            self.wrapped_module.load_state_dict(state_dict, strict)
        except RuntimeError as e:
            if not suppress_autoconvert_weights and 'Missing key(s) in state_dict' in e.__str__():
                print("SwitchedConvConversionWrapper.load_state_dict: Automatically converting provided weights for use with switched_conv. Note: all state dict mismatch errors will be suppressed! Be absolutely sure this state_dict is the one you want!")
                converted = convert_state_dict_to_switched_conv(state_dict, self.breadth, self.allow_list)
                self.wrapped_module.load_state_dict(converted, strict=False)  # strict=False required because coupler parameters are not converted and will not be in the converted state dict.
            else:
                raise e

    # Allow clients to pass through this wrapper to access the wrapped module.
    def __getattr__(self, item):
        if item != 'wrapped_module' and item != 'load_state_dict' and hasattr(self.wrapped_module, item):
            return getattr(self.wrapped_module, item)
        raise AttributeError(f"Requested attribute {item} does not exist in SwitchedConvConversionWrapper or wrapped object.")

    def __call__(self, *args, **kwargs):
        return self.wrapped_module(*args, **kwargs)


if __name__ == '__main__':
    convert_state_dict_to_switched_conv('work_dirs/deeplabv3_r50-d8_512x1024_80k_cityscapes_20200606_113404-b92cfdd4.pth', 8,
                                        ['decode_head.bottleneck.conv', 'auxiliary_head.convs.0.conv', 'backbone.layer1.0.downsample.0', 'backbone.layer2.0.downsample.0'])