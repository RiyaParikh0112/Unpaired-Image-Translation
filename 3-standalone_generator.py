'''
The ResNet architecture allows for the training of very deep neural networks by introducing residual connections that skip some layers,
which helps prevent the vanishing gradient problem.
We first define the ResNet Blocks. These are blocks compromised of two 3X3 CNN layers where the input to the block is concatenated
to the output of the block, channel-wise.
This function creates 2 Convulation-InstanceNorm blocks with 3X3 filter and 1X1 stride.
Same padding is used instead of refelction padded reccomended in the paper.
'''
def resnet_block(n_filters,input_layer):
    init = RandomNormal(stddev=0.02)
    #1 convolution layer
    g = Conv2D(n_filters,(3,3),padding='same',kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    #2 convolution layer
    g = Conv2D(n_filters,(3,3),padding='same',kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    #concatenate merge channel-wise with the input layer
    g = Concatenate()([g,input_layer])
    return g

'''define a function that creates 9-ResNet block version for 256X256 input images. Importantly, the model outputs pixel values with the
shape as the input and pixel values are in the range [-1,1]

'''

def define_generator(image_shape,n_resnet=9):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    #c7s1-64 7x7 filter and stride of 1, with 64 output channels.
    g = Conv2D(64,(7,7),padding='same',kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    #d128
    g = Conv2D(128,(3,3),strides=(2,2),padding='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    #d256
    g = Conv2D(256,(3,3),strides=(2,2),padding='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    #R256 ResNet block with 256 output channels.
    for _ in range(n_resnet):
        g = resnet_block(256,g)
    '''Apply a series of transposed convolutional layers with Instance Normalization and ReLU activation,
    before outputting the generated image with the same shape as the input image.'''
    #u128 transposed convolutional layer with 128 output channels, and a stride of 2, which corresponds to upsampling the feature map
    g = Conv2DTranspose(128,(3,3),strides=(2,2),padding='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    #u64
    g = Conv2DTranspose(64,(3,3),strides=(2,2),padding='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    #c7s1-3
    g = Conv2DTranspose(3,(7,7),padding='same',kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tan_h')(g)
    '''tan_h activation function is used for the output image to ensure pixel values are in the range of [-1,1].'''
    #define model
    model = Model(in_image,out_image)
    return model
    
    
    
