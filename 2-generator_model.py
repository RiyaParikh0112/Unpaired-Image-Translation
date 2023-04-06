'''This function implements the 70X70 patchGAN discriminator model. 
The model takes 256X256 sized image as input and outputs a patch of perdictions.
the model is optimized using least square loss (implemented as mean squared error), and a weighting is used so that the updates if the model 
have half (0.5) the usual effect.
(this was reccomended by the authors of the paper to slow down the changes to the discriminator relative to the generator model during training)'''

def define_discriminator(image_shape):
    #weight initialization
    init = RandomNormal(stddev=0.02)
    #source image input
    in_image = Input(shape=image_shape)
    #64
    d = Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    #128
    d = Conv2D(128,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    #256
    d = Conv2D(256,(4,4),strides=(2,2),padding='same',kernel_initalizer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyRelu(alpha=0.2)(d)
    #512
    d = Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)
    d =InstanceNormalization(axis=-1)(d)
    d = LeakyRelu(alpha=0.2)(d)
    #prefinal output layer
    d = Conv2D(512,(4,4),padding='same',kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyRelu(alpha=0.2)(d)
    #patch output
    patch_out = Conv2D(1,(4,4),padding='same',kernel_initializer=init)(d)
    #definig the model
    model = Model(in_image,patch_out)
    #compiling the model
    model.compile(loss = 'mse',optimizer = Adam(lr=0.0002,beta_1=0.5),loss_weights=[0.5])
    return model
