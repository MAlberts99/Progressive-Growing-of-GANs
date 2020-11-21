import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Small container to hold the weight and bias parameters for the individual modules
# Implements bias zero initialisation and He runtime initialisation
class Mini_Container(nn.Module):
  def __init__(self, shape, bias=False, bias_init=0, gain=2.0, use_wscale = False, fan_in=None):
    super().__init__()
    self.use_wscale = use_wscale

    if fan_in is None:  fan_in = np.prod(shape[1:]) # He initialisation
    self.std = torch.tensor(np.sqrt(gain/fan_in))

    if use_wscale: # Make Parameter
      self.weight = nn.Parameter(torch.randn(*shape))
    else:
      self.weight = nn.Parameter(torch.empty(*shape).normal_(mean=0, std=1))

    if bias:
      self.bias = nn.Parameter(torch.empty(shape[0]).fill_(bias_init))
    else:
      self.bias = None

  def get_weight(self):
    if self.use_wscale:
      return self.weight, self.std
    return self.weight, 1.0

  def get_bias(self):
    return self.bias

  def __repr___(self):
      return str(self.weight.shape) + " " + str(bias.shape)

  def forward(self):
    return True


#### Base Layers ####
# Linear layer using the above Mini_Container
class linear(nn.Module):
  def __init__(self, fmaps_in, fmaps_out, bias=True, gain=2.0, use_wscale=False):
    super().__init__()
    self.params = Mini_Container([fmaps_out, fmaps_in], bias=bias, gain=gain, use_wscale=use_wscale)

  def forward(self, x):
    weight, mult = self.params.get_weight()
    bias = self.params.get_bias()
    out = F.linear(x, weight, bias=bias) * mult
    return out

# Conv2d Layer implementing the Mini_Container
class conv2d(nn.Module):
  def __init__(self, fmaps_in, fmaps_out, kernel_size, padding=True, bias=True, gain=2.0, use_wscale=False):
    super().__init__()
    self.params = Mini_Container([fmaps_out, fmaps_in, kernel_size, kernel_size], bias=bias, gain=gain, use_wscale=use_wscale)

    if padding:
      self.padding = kernel_size//2
    else:
      self.padding = 0
    self.stride = 1

  def forward(self, x):
    weight, mult = self.params.get_weight()
    bias = self.params.get_bias()
    out = F.conv2d(x, weight, bias=bias, padding=self.padding) * mult
    return out


#### Normalisation and Helper Layers ####
# Pixel Norm
def Pixel_Norm(x):
  norm = torch.rsqrt(torch.mean(torch.square(x), axis=1, keepdim=True) + 1e-8)
  return x * norm

# Mini Batch Standard Deviation Adapted from:
def miniBatchStdDev(x, subGroupSize=4):

    size = x.size()
    subGroupSize = min(size[0], subGroupSize)
    if size[0] % subGroupSize != 0:
        subGroupSize = size[0]
    G = int(size[0] / subGroupSize)
    if subGroupSize > 1:
        y = x.view(-1, subGroupSize, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, subGroupSize, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    return torch.cat([x, y], dim=1)


# Reshape to 4x4
def Reshape(x, fmaps):
  return x.view(x.shape[0], fmaps, 4, 4)

 #### Combination Layers ####

 # Convolution, Activation and Normalisation. Used in the Generator
class Fused_Conv_Act_Norm(nn.Module):
  def __init__(self, fmaps_in, fmaps_out, kernel_size=3, negative_slope=0.2, padding=True, use_norm=True, use_wscale=False):
    super().__init__()
    self.conv = conv2d(fmaps_in, fmaps_out, kernel_size=kernel_size, padding=padding, use_wscale=use_wscale)

  def forward(self, x):
    return Pixel_Norm(F.leaky_relu(self.conv(x), negative_slope=0.2))

# Convolution and Activation, no Normalisation. Used in the Discriminator
class Fused_Conv_Act(nn.Module):
  def __init__(self, fmaps_in, fmaps_out, kernel_size=3, negative_slope=0.2, padding=True, use_norm=True, use_wscale=False):
    super().__init__()
    self.conv = conv2d(fmaps_in, fmaps_out, kernel_size=kernel_size, padding=padding, use_wscale=use_wscale)

  def forward(self, x):
    return F.leaky_relu(self.conv(x), negative_slope=0.2)
