# Progressive-Growing-of-GANs
This project is an unofficial PyTorch implementation of the paper using Google Colab: [Progressive Growing of GANs for Improved Quality, Stability, and Variation
](https://arxiv.org/abs/1710.10196)

All credit goes to: Tero Karras, Timo Aila, Samuli Laine and Jaakko Lehtinen

**Careful**: This project is not finished. 
## Description
The paper describes a new way of training GANs which greatly increases the image quality. The key difference is that the GAN is grown progressively, meaning that lower resolution layers are trained first, with subsequent higher dimensional ones being faded in as training progresses. For example the training starts with a 16x16 generator and discriminator (a in the image below). After a set amount of training time the next higher layer, e.g. 32x32, is slowly faded in (b). This proceeds by the previous and new layer both producing an output. At first during the "fading-in" period the output of the new layer has a very low weighting (alpha) which increases as the "fading-in" progresses. This is then followed by a stabilisation period during which the output of the new layer is evaluated (c). This is illustrated in the image below.

<p align='center'>
  <img src='Images/Progressive growing.png' width="600px">
</p>
<em>Adapted from: Progressive Growing of GANs for Improved Quality, Stability, and Variation</em>

## Current State of the Project
This project is currently still under development.

What has been implemented:
- Pixelwise Normalisation
- Minibatch Standard Deviation
- Equalised Learning Rate
- Generator 
- Discriminator
- Wasserstein GP loss with extra term
- Progressive GAN trainer

What is still missing:
- Exponential moving average of the weights

With the above implemented the GAN would work. All generator and discriminator specific features (pixelnorm, minibatch std and equalised learning rate) have been implemented and work. The generator and discriminator have been tested using the provided pretrained weights. Both work. However, when training using the trainer the GAN diverges.
