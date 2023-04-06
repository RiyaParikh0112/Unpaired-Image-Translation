
# Image to Image translation using CycleGAN 🌌

In this project we will create deep convulational neural networks for image-to-image translation tasks.
Unlike other GANs, CycleGAN does not require a dataset of paired images.

## CycleGAN ♼
The code was implemented after taking reference from the Paper by Jan-Yan Zhu in their 2017 paper titled [Unpaired Image-to-Image Translation using Cycle-Consistent Adversial Networks](https://arxiv.org/abs/1703.10593). 

## Model Architecture 𝌭

The model Architecture is compromised of two generator models.
1. One generator (Generator-A) for training images for the first domain.(Domain-A)
2. Second generator(Generator-B)for generating images for the second domain (Domain-B)

The generator model performs **Image translation**

Domain A -> Generator B -> Domain B \
Domain B -> Generator A -> Domain A 

- There is a corresponding discriminator model for every generator.

The first discriminator model (Discriminator-A) takes real images form Domain-A and generated images from Generator-A and **predict whether they are real/fake** and likewise from Discriminator-B.

- Domain-A -> Discriminator-A -> [Real/Fake]
- Domain-B -> Generator-A -> Discriminator-A -> [Real/Fake]
- Domain-B -> Discriminator-B -> [Real/Fake]
- Domain-A -> Generator-B -> Discriminator-B -> [Real/Fake]

**Training is done in adversial zero-sum process** which means the generator learn to better fool the discriminator and the discriminator learns to better detect the fake images.\

The generator models are regularized not just to create new images in target domain, but instead translate more reconstructed versions of input images from source domain. This is acheived by using generated images as input to the corresponding generator model and comparing the output image to the original images. **Passing an image through both the generators is called Cycle.** Together each pair od generator models are trained to better produce the original source image, reffered to as *cycle consistency*.

- Domain-B -> Generator-A -> Domain-A -> Generator-B ->Domain-B
- Domain-A -> Generator-B -> Domain-B -> Generator-A -> Domain-A

Next step in the Architecture is **identity mapping**. In this step the generator is provided with inputs from the target domain and is expected to *generate the same image without change*. This step of the architecture is not a compulsion but this results in better matching of the color profile of the input image.

- Domain-A -> Generator-A -> Domain-A
- Domain-B -> Generator-B -> Domain-B

![Simplified Architecture of CycleGAN](https://i.ibb.co/BVDkhVV/Screenshot-2023-04-06-at-7-40-10-AM.png)
