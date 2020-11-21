# Progressive-Growing-of-GANs
This project is an unofficial PyTorch implementation of the paper using Google Colab: [Progressive Growing of GANs for Improved Quality, Stability, and Variation
](https://arxiv.org/abs/1710.10196)

All credit goes to: Tero Karras, Timo Aila, Samuli Laine and Jaakko Lehtinen

**Careful**: This project is not finished. 
## Description
The paper describes a new way of training GANs which greatly increases the image quality. The key difference is that the GAN is grown progressively, meaning that lower resolution layers are trained first, with subsequent higher dimensional ones being faded in as training progresses. For example the training starts with a 4x4 generator and discriminator. After a set amount of training time the next higher layer, e.g. 8x8, is slowly faded in. This proceeds by the previous and new layer both producing an output. At first during the "fading-in" period the output of the new layer has a very low weighting which increases as the "fading-in" progresses. This is then followed by a stabilisation period during which the output of the new layer is evaluated. This is illustrated in the image below.

<p align='center'>
  <img src='Images/Progressive growing.png' width="600px">
</p>
<em>Adapted from: Progressive Growing of GANs for Improved Quality, Stability, and Variation</em>
In the above image stabilisation training is illustrated in a and c. In b the new layer is faded in.

## Current State of the Project

