import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

# Takes a feature map coming from the network and converts it to an n channel image
class toRGB(nn.Module):
  def __init__(self, fmaps, out_dim=3, use_wscale=False):
    super().__init__()
    self.conv = conv2d(fmaps, out_dim, kernel_size=1, gain=1, use_wscale=use_wscale)

  def forward(self, x):
    return self.conv(x)

# The first Block of the Generator, similar to normal Block but with linear
# layer for the latent dimension
class First_Gen_Block(nn.Module):
  def __init__(self, latent_dim, fmaps, out_dim=3, normalise_latent=True, use_wscale=False):
    super().__init__()
    self.normalise_latent = normalise_latent # wether or not to apply Pixel Norm to latent
    self.fmaps = fmaps
    self.latent_dim = latent_dim
    self.out_dim = out_dim

    self.l1 = linear(fmaps_in=latent_dim, fmaps_out=fmaps*16, gain=2.0, use_wscale=use_wscale) # Dense layer for converting latent to 4x4
    self.conv1 = Fused_Conv_Act_Norm(fmaps, fmaps, kernel_size=3, use_wscale=True)
    self.rgb = toRGB(fmaps, out_dim)

  def extra_rep(self):
    print("First Gen Block, latent_dim: {}, fmaps_out: {}, out_dim: {}, normalise_latent: {}".format(self.latent_dim, self.fmaps, self.out_dim, self.normalise_latent))

  def forward(self, x, last=False, growing=False): # Last: wether toRGB is applied or not
    if self.normalise_latent:
      x = Pixel_Norm(x)
    x = F.leaky_relu(self.l1(x), negative_slope=0.2) # linear, leaky ReLU activation, PixelNorm
    x = Pixel_Norm(Reshape(x, self.fmaps))
    x = self.conv1(x)

    if last:
      x_rgb = self.rgb(x)
      if growing:
        return x, x_rgb
      else:
        return x_rgb
    return x

# Generator Block, contains two Convolution Blocks
class Gen_Block(nn.Module):
  def __init__(self, fmaps_in, fmaps_out, out_dim=3, use_wscale=False):
    super().__init__()
    self.fmaps_in = fmaps_in
    self.fmaps_out = fmaps_out
    self.out_dim = out_dim

    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    self.conv1 = Fused_Conv_Act_Norm(fmaps_in, fmaps_out, kernel_size=3, use_wscale=use_wscale)
    self.conv2 = Fused_Conv_Act_Norm(fmaps_out, fmaps_out, kernel_size=3, use_wscale=use_wscale)

    self.rgb = toRGB(fmaps_out, out_dim)

  def extra_rep(self):
    print("Gen Block, fmaps_in: {}, fmaps_out: {}, out_dim: {}".format(self.fmaps_in, self.fmaps_out, self.out_dim))

  def forward(self, x, last=False, growing=False): # Growing: if the network grows from this layer
    x = self.conv2(self.conv1(self.upsample(x)))

    if last:
      x_rgb = self.rgb(x)
      if growing:
        return x, x_rgb
      else:
        return x_rgb
    return x


class Generator(nn.Module):

  def __init__(self, latent_dim=512, fmaps_max=512, out_dim=3, normalise_latent=True, out_res=None, use_wscale=True):
    super().__init__()
    assert out_res in [4, 8, 16, 32, 64, 128, 256, 512, 1024, None]
    res_dict = {4:1,
                8:2,
                16:3,
                32:4,
                64:5,
                128:6,
                256:7,
                512:8,
                1024:9}

    self.out_dim = out_dim
    self.use_wscale = use_wscale
    self.fmaps_base = 8
    self.fmaps_max = fmaps_max
    self.latent_dim = latent_dim
    self.layers = res_dict[out_res]
    self.current_layer = 1 # Because first layer has been added


    self.blocks = nn.ModuleList()
    self.blocks.append(First_Gen_Block(latent_dim, self.calc_fmaps(0), out_dim=out_dim, normalise_latent=normalise_latent, use_wscale=use_wscale))

    if out_res:
        self.built_model()


  ### Building Functions ###
  def calc_fmaps(self, layer): return min(self.fmaps_base*2**(9-layer), self.fmaps_max)

  def built_model(self):
    for i in range(1, self.layers):
      self.add_block()
      print(self.calc_fmaps(i-1), self.calc_fmaps(i))

  def add_block(self):
    assert self.current_layer < self.layers
    self.blocks.append(Gen_Block(self.calc_fmaps(self.current_layer-1),
                                 self.calc_fmaps(self.current_layer), out_dim=self.out_dim, use_wscale=self.use_wscale))

    self.current_layer += 1

  def repr(self):
      for block in self.blocks:
          block.extra_rep()

  ### Forwarding ###
  def growing(self, x, alpha=1.0):
    assert alpha <= 1.0 and alpha >= 0.0

    out_old = x
    for i, block in enumerate(self.blocks[:-2]):
      out_old = block(out_old, last=False)

    out_old, out_old_rgb = self.blocks[len(self.blocks)-2](out_old, last=True, growing=True)
    out_new_rgb = self.blocks[len(self.blocks)-1](out_old, last=True)

    out_old_rgb = F.interpolate(out_old_rgb, scale_factor=2, mode="nearest")

    out_comb = (1-alpha) * out_old_rgb + alpha * out_new_rgb
    return out_comb

  def normal_forward(self, x):
    out = x
    for i, block in enumerate(self.blocks):
      if i is (len(self.blocks)-1):
        out = block(out, True)
      else:
        out = block(out)
    return out

  def forward(self, x, growing = False, alpha=1.0):
    if growing:
      return self.growing(x, alpha)
    return self.normal_forward(x)
