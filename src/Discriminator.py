import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from .utils import *

class fromRGB(nn.Module):
  def __init__(self, fmaps_out, in_dim = 3, use_wscale=False):
    super().__init__()
    self.conv = conv2d(in_dim, fmaps_out, kernel_size=1, gain=2.0, use_wscale=use_wscale)

  def forward(self, x):
    return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Last_Disc_Block(nn.Module):
  def __init__(self, fmaps, in_dim=3, minibatch_std=True, use_wscale=False):
    super().__init__()
    self.fmaps = fmaps
    self.in_dim = in_dim
    self.minibatch_std = minibatch_std

    if minibatch_std:
      self.conv1 = Fused_Conv_Act(fmaps+1, fmaps, kernel_size=3, use_wscale=use_wscale)
    else:
      self.conv1 = Fused_Conv_Act(fmaps, fmaps, kernel_size=3, use_wscale=use_wscale)

    self.conv2 = Fused_Conv_Act(fmaps, fmaps, kernel_size=4, padding = False, use_wscale=use_wscale)
    self.linear1 = linear(fmaps, 1, gain = 2.0, use_wscale=use_wscale)

    self.rgb = fromRGB(fmaps, in_dim)

  def extra_repr(self):
    print("Last Disc Block, fmaps: {}, in_dim: {}, minibatch_std:{}".format(self.fmaps, self.in_dim, self.minibatch_std))

  def forward(self, x, first=False, growing=False): # Growing: if the network grows from this layer
    if first:
      x = self.rgb(x)
    if growing:
      return x
    if self.minibatch_std:
      x = miniBatchStdDev(x)

    x = self.conv2(self.conv1(x)) # 2 Convolution Layers
    x = self.linear1(x.view(x.shape[0], -1)) # Flatten then linear
    return x

class Disc_Block(nn.Module):
  def __init__(self, fmaps_in, fmaps_out, in_dim=3, use_wscale=False):
    super().__init__()
    self.fmaps_in = fmaps_in
    self.fmaps_out = fmaps_out
    self.in_dim = in_dim

    self.conv1 = Fused_Conv_Act_Norm(fmaps_in, fmaps_out, kernel_size=3, use_wscale=use_wscale)
    self.conv2 = Fused_Conv_Act_Norm(fmaps_out, fmaps_out, kernel_size=3, use_wscale=use_wscale)
    self.down = nn.AvgPool2d((2, 2))

    self.rgb = fromRGB(fmaps_in, in_dim)

  def extra_repr(self):
    print("Disc Block, fmaps_in: {}, self.fmaps_out: {}, in_dim: {}".format(self.fmaps_in, self.fmaps_out, self.in_dim))

  def forward(self, x, first=False, growing=False): # Growing: if the network grows from this layer
    if first:
      x = self.rgb(x)
    if growing:
      return x
    x = self.down(self.conv2(self.conv1(x))) # Conv1, Conv2 and then Downsample
    return x

class Discriminator(nn.Module):

  def __init__(self, fmaps_max=512, in_dim=3, minibatch_std = True, in_res = None, use_wscale=True):
    super().__init__()
    assert in_res in [4, 8, 16, 32, 64, 128, 256, 512, 1024, None]

    self.in_dim = in_dim
    self.use_wscale = use_wscale
    self.fmaps_base = 8
    self.fmaps_max = fmaps_max
    self.current_layer = 1


    self.blocks = nn.ModuleList()
    self.blocks.append(Last_Disc_Block(self.calc_fmaps(0), in_dim=self.in_dim, minibatch_std=minibatch_std, use_wscale=use_wscale))

    if in_res:
      self.built_model(in_res)

  ### Building Functions ###
  def calc_fmaps(self, layer): return min(self.fmaps_base*2**(9-layer), self.fmaps_max)

  def repr(self):
      for block in self.blocks:
          block.extra_repr()

  def built_model(self, resolution):
    layers = int(math.log2(resolution)-1)

    for i in range(1, layers):
      self.add_block()
      print("Adding Layer: ", self.calc_fmaps(i), self.calc_fmaps(i-1))


  def add_block(self):
    self.blocks.append(Disc_Block(self.calc_fmaps(self.current_layer),
                                  self.calc_fmaps(self.current_layer-1), in_dim=self.in_dim, use_wscale=self.use_wscale))
    self.current_layer +=1

  ### Forwarding ###
  def growing(self, x, alpha=1.0):
    assert alpha <= 1.0 and alpha >= 0.0
    out_new = self.blocks[len(self.blocks)-1](x, first=True)

    out_old = F.avg_pool2d(x, (2, 2))
    out_old = self.blocks[len(self.blocks)-2](out_old, first=True, growing=True) # Run fromRGB over the downsampled input

    out_comb = (1-alpha) * out_old + alpha * out_new

    for i, block in enumerate(reversed(self.blocks[:-1])):
      out_comb = block(out_comb, first=False, growing=False)
    return out_comb

  def normal_forward(self, x):
    out = x
    for i, block in enumerate(reversed(self.blocks)):
      if i is 0:
        out = block(out, first=True)
      else:
        out = block(out)
    return out

  def forward(self, x, growing=False, alpha=1.0):
    assert x.shape[2] == x.shape[3]
    if growing:
      return self.growing(x, alpha)
    return self.normal_forward(x)
