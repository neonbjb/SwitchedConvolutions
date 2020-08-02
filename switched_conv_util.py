import torch
from torch import nn
import torchvision
import os
import torch.nn.init as init
from matplotlib import cm
import numpy as np


# Universal weight initialization procedure used by switched_conv
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                m.bias.data.zero_()


# Copied from torchvision.utils.save_image. Allows specifying pixel format.
def save_image(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None, pix_format=None):
    from PIL import Image
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                                       normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr, mode=pix_format).convert('RGB')
    im.save(fp, format=format)


def save_attention_to_image(folder, attention_out, attention_size, step, fname_part="map", l_mult=1.0):
    magnitude, indices = torch.topk(attention_out, 1, dim=-1)
    magnitude = magnitude.squeeze(3)
    indices = indices.squeeze(3)
    # indices is an integer tensor (b,w,h) where values are on the range [0,attention_size]
    # magnitude is a float tensor (b,w,h) [0,1] representing the magnitude of that attention.
    # Use HSV colorspace to show this. Hue is mapped to the indices, Lightness is mapped to intensity,
    # Saturation is left fixed.
    hue = indices.float() / attention_size
    saturation = torch.full_like(hue, .8)
    value = magnitude * l_mult
    hsv_img = torch.stack([hue, saturation, value], dim=1)

    output_path=os.path.join(folder, "attention_maps", fname_part)
    os.makedirs(output_path, exist_ok=True)
    save_image(hsv_img, os.path.join(output_path, "attention_map_%i.png" % (step,)), pix_format="HSV")


def save_attention_to_image_rgb(output_folder, attention_out, attention_size, file_prefix, step, cmap_discrete_name='viridis'):
    magnitude, indices = torch.topk(attention_out, 3, dim=-1)
    magnitude = magnitude.cpu()
    indices = indices.cpu()
    magnitude /= torch.max(torch.abs(torch.min(magnitude)), torch.abs(torch.max(magnitude)))
    colormap = cm.get_cmap(cmap_discrete_name, attention_size)
    colormap_mag = cm.get_cmap(cmap_discrete_name)
    os.makedirs(os.path.join(output_folder), exist_ok=True)
    for i in range(3):
        img = torch.tensor(colormap(indices[:,:,:,i].numpy()))
        img = img.permute((0, 3, 1, 2))
        save_image(img, os.path.join(output_folder, file_prefix + "_%i_%s.png" % (step, "rgb_%i" % (i,))), pix_format="RGBA")

        mag_image = torch.tensor(colormap_mag(magnitude[:,:,:,i].numpy()))
        mag_image = mag_image.permute((0, 3, 1, 2))
        save_image(mag_image, os.path.join(output_folder, file_prefix + "_%i_%s.png" % (step, "mag_%i" % (i,))), pix_format="RGBA")

if __name__ == "__main__":
    adata = torch.randn(12, 64, 64, 8)
    save_attention_to_image_rgb(".", adata, 8, "pre", 0)