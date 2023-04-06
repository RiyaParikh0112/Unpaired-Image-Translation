
# Image to Image translation using CycleGAN

In this project we will create deep convulational neural networks for image-to-image translation tasks.
Unlike other GANs, CycleGAN does not require a dataset of paired images.

## CycleGAN ♼
The code was implemented after taking reference from the Paper by Jan-Yan Zhu in their 2017 paper titled [Unpaired Image-to-Image Translation using Cycle-Consistent Adversial Networks](https://arxiv.org/abs/1703.10593). 

## Model Architecture

The model Architecture is compromised of two generator models.
1. One generator (Generator-A) for training images for the first domain.(Domain-A)
2. Second generator(Generator-B)for generating images for the second domain (Domain-B)
