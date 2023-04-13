
#  Image to Image translation using CycleGAN🌌

In this project we will create deep convulational neural networks for image-to-image translation tasks.
Unlike other GANs, CycleGAN does not require a dataset of paired images.\
We want to take an image from an input domain  Di
  and then transform it into an image of target domain  Dt
  without necessarily having a one-to-one mapping between images from input to target domain in the training set. Relaxation of having one-to-one mapping makes this formulation quite powerful - the same method could be used to tackle a variety of problems by varying the input-output domain pairs - performing artistic style transfer, adding bokeh effect to phone camera photos, creating outline maps from satellite images or convert horses to zebras and vice versa.\
 <img src="Images/output paper examples.png" alt="Model" width="60%" />

## CycleGAN ♼
The code was implemented after taking reference from the Paper by Jan-Yan Zhu in their 2017 paper titled [Unpaired Image-to-Image Translation using Cycle-Consistent Adversial Networks](https://arxiv.org/abs/1703.10593). \
*The need for a paired image in the target domain is eliminated by making a two-step transformation of source domain image - first by trying to map it to target domain and then back to the original image. Mapping the image to target domain is done using a generator network and the quality of this generated image is improved by pitching the generator against a discrimintor* 

<img src='Images/paired_unpaired.jpeg' alt='paired_unpaired' width='40%'/>

## Cycle Consistency
`Adversarial training can, in theory, learn mappings  G
  and  F
  that produce outputs identically distributed as target domains  Y
  and  X
  respectively. However, with large enough capacity, a network can map the same set of input images to any random permutation of images in the target domain, where any of the learned mappings can induce an output distribution that matches the target distribution. Thus, an adversarial loss alone cannot guarantee that the learned function can map an individual input  xi
  to a desired output  yi
 .`
 To regularize the model, the authors introduce the constraint of cycle-consistency - if we transform from source distribution to target and then back again to source distribution, we should get samples from our source distribution.


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

*the generator and discriminator are actually playing a game whose Nash equilibrium is achieved when the generator's distribution becomes same as the desired distribution.*

![Simplified Architecture of CycleGAN](https://i.ibb.co/BVDkhVV/Screenshot-2023-04-06-at-7-40-10-AM.png)

## 1. Discriminator
The discriminator is a deep convolutional neural network that performs image classification. It takes a source image as input and predicts the likelihood of whether the target image is a real or fake image. Two discriminator models are used, one for Domain-A (scenary) and one for Domain-B (van gogh).

The discriminator design is based on the effective receptive field of the model, which defines the relationship between one output of the model to the number of pixels in the input image. This is called a PatchGAN model and is carefully designed so that each output prediction of the model maps to a 70×70 square or patch of the input image. The benefit of this approach is that the same model can be applied to input images of different sizes, e.g. larger or smaller than 256×256 pixels.

The output of the model depends on the size of the input image but may be one value or a square activation map of values. Each value is a probability for the likelihood that a patch in the input image is real. These values can be averaged to give an overall likelihood or classification score if needed.

<img src='Images/discriminator.jpeg' width = '80%'/>

## 2. Generator 🥷

The generator is an encoder-decoder model architecture. The model takes a source image (e.g. scenary photo) and generates a target image (e.g. van gogh photo). It does this by first downsampling or encoding the input image down to a bottleneck layer, then interpreting the encoding with a number of ResNet layers that use skip connections, followed by a series of layers that upsample or decode the representation to the size of the output image.\
we can define a function that will create the 9-resnet block version for 256×256 input images. This can easily be changed to the 6-resnet block version by setting image_shape to (128x128x3) and n_resnet function argument to 6.

<img src='Images/Generator.jpeg' width = '80%'>

<img src='Images/Resnet.jpeg' width = '40%'>

## 3. Composite Model
Altogether, each generator model is optimized via the combination of four outputs with four loss functions:

- Adversarial loss (L2 or mean squared error).
- Identity loss (L1 or mean absolute error).
- Forward cycle loss (L1 or mean absolute error).
- Backward cycle loss (L1 or mean absolute error).

This can be achieved by defining a composite model used to train each generator model that is responsible for only updating the weights of that generator model, although it is required to share the weights with the related discriminator model and the other generator model.

This is implemented in the define_composite_model() function below that takes a defined generator model (g_model_1) as well as the defined discriminator model for the generator models output (d_model) and the other generator model (g_model_2). The weights of the other models are marked as not trainable as we are only interested in updating the first generator model, i.e. the focus of this composite model.

The discriminator is connected to the output of the generator in order to classify generated images as real or fake. A second input for the composite model is defined as an image from the target domain (instead of the source domain), which the generator is expected to output without translation for the identity mapping. Next, forward cycle loss involves connecting the output of the generator to the other generator, which will reconstruct the source image. Finally, the backward cycle loss involves the image from the target domain used for the identity mapping that is also passed through the other generator whose output is connected to our main generator as input and outputs a reconstructed version of that image from the target domain.

To summarize, a composite model has two inputs for the real photos from Domain-A and Domain-B, and four outputs for the discriminator output, identity generated image, forward cycle generated image, and backward cycle generated image.

**Generator-A Composite Model (BtoA or scenary to painting)**

The inputs, transformations, and outputs of the model are as follows:

- Adversarial Loss: Domain-B -> Generator-A -> Domain-A -> Discriminator-A -> [real/fake]
- Identity Loss: Domain-A -> Generator-A -> Domain-A
- Forward Cycle Loss: Domain-B -> Generator-A -> Domain-A -> Generator-B -> Domain-B
- Backward Cycle Loss: Domain-A -> Generator-B -> Domain-B -> Generator-A -> Domain-A

We can summarize the inputs and outputs as:

- Inputs: Domain-B, Domain-A
- Outputs: Real, Domain-A, Domain-B, Domain-A
- Generator-B Composite Model (AtoB or Horse to Zebra)

The inputs, transformations, and outputs of the model are as follows:

- Adversarial Loss: Domain-A -> Generator-B -> Domain-B -> Discriminator-B -> [real/fake]
- Identity Loss: Domain-B -> Generator-B -> Domain-B
- Forward Cycle Loss: Domain-A -> Generator-B -> Domain-B -> Generator-A -> Domain-A
- Backward Cycle Loss: Domain-B -> Generator-A -> Domain-A -> Generator-B -> Domain-B

We can summarize the inputs and outputs as:

- Inputs: Domain-A, Domain-B
- Outputs: Real, Domain-B, Domain-A, Domain-B



## Screenshots
![Model](https://i.ibb.co/Y3ykG0H/Screenshot-2023-04-11-at-9-59-57-AM.png)
