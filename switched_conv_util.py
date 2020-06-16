import torch
import torchvision
import os

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


def save_attention_to_image(attention_out, attention_size, step, fname_part="map", l_mult=1.0):
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

    os.makedirs("attention_maps/%s" % (fname_part,), exist_ok=True)
    save_image(hsv_img, "attention_maps/%s/attention_map_%i.png" % (fname_part, step,), pix_format="HSV")