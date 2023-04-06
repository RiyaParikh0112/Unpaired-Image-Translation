'''In the composite model the weights of the other model are marked as not trainable as we are interested in updating the first generator
model,i.e focus in the composite model
- discriminator is connected to the output of the generator to classify the images as real or fake.
-  second input for the composite model is defined as an image from the target domain(instead of the source domain),which the generator is 
expected to output without the translation for the identity mapping.
- Forward Cycle Loss involves connecting the output of the generator to the other generator,which will reconstruct source image.
- Backward Cycle loss involves the image from the target domain used for the identity mapping then is also passed through the other 
generator whose output is connected to our main generator as input and outputs a reconstructed version of that image from a target domain.

Hence, 2 inputs and 4 outputs.

Only weights of the first or main generator models are updated for the composite model as this is done by weighted sum of all loss 
functions.
Cycle loss has more weight than adverserial loss in the paper(10 times)
'''
def define_composite_model(g_model_1,d_model,g_model_2,image_shape):
    #ensure the model we are updating is trainable
    g_model_1.trainable = True
    #mark discriminator as non-trainable
    d_model.trainable = False
    g_model_2.trainable = False
    #discriminator element
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    #identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    #forward cycle
    output_f = g_model_2(gen1_out)
    #backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    #define the model graph
    model = Model([input_gen,input_id],[output_d,output_id,output_f,output_b])
    #optimization algorithm
    opt = Adam(lr=0.0002,beta_1=0.5)
    #model compile
    model.compile(loss=['mse','mae','mae','mae'],loss_weights=[1,5,10,10],optimizer=opt)
    return model
