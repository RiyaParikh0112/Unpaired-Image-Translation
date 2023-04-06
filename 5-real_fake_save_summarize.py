'''Load paired images dataset in Numpy array format, it will return list of 2 numpy arrays'''
def load_real_samples(filename):
    data = load(filename)
    #unpack the arrays
    X1,X2 = data['arr_0'],data['arr_1']
    #scaling from [0,255] to [-1,1]
    X1 = (X1-127.5)/127.5
    X2 = (X2 - 127.5)/127.5
    return X1,X2

#selecting random batch of images for every iteration
'''This is implemented by taking the np array for a domain as input and returning the requested number of randomly selected
images, as well as the target for the PatchGAN dicriminator model indicating the images are real(target_value = 1.0)
'''

def generate_real_samples(dataset,n_samples,patch_shape):
    ix = randint(0,dataset.shape[0],n_samples)
    #retreive selected images
    X = dataset[ix]
    # generate 'real class labels (1)'
    y = ones((n_samples,patch_shape,patch_shape,1)) #this is 16X16X1 activation map
    return X,y

def generate_fake_samples(g_model,dataset,patch_shape):
    X = g_model.predict(dataset)
    y = zeros((len(X),patch_shape,patch_shape,1))
    return X,y

'''GAN models dont converge, it finds an equilibrium in generator and discriminator, so we cannot judge when the training should stop.
So we save the model and use to it to generate sample image to image translations periodically during training
'''
def save_models(step,g_model_AtoB,g_model_BtoA):
    filename1 = 'g_model_AtoB_%06d.h5'%(step+1)
    g_model_AtoB.save(filename1)
    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('...Saved: %s and %s'%(filename1,filename2))
    
# this function uses given generator model to generate translated versions of randomly selected images and saves the plot to file
def summarize_performance(step,g_model,trainX,name,n_samples=5):
    #select sample of input images
    X_in, _ = generate_real_samples(trainX,n_samples,0)
    #generate translated images
    X_out,_ = generate_fake_samples(g_model,X_in,0)
    #scale all pixels from [-1,1] to [0,1]
    X_in = (X_in +1)/2.0
    X_out = (X_out +1)/2.0
    #plot real images
    for i in range(n_samples):
        pyplot.subplot(2,n_samples,1+i)
        pyplot.axis('off')
        pyplot.imshow(X_in[i])
    #plot for translated imafes
    for i in range(n_samples):
        pyplot.subplot(2,n_samples,1+n_samples+i)
        pyplot.axis('off')
        pyplot.imshow(X_out[i])
    #save the plot to file
    filename1 = '%s_generated_plot_%06d.png'%(name,(step=1))
    pyplot.savefig(filename1)
    pyplot.close()
    
# to manage how quickly discriminator model learns a pool of fake images is maintained
'''In the paper they define an image pool of 50 generated images for each discriminator model that is first populated and
probabilistically either adds new images to the pool by replacing an exisitng images or uses a generated image directly.
'''
def update_image_pool(pool,images,max_size=50):
    selected = list()
    for image in images:
        if len(pool)<max_size:
            #stock the pool
            pool.append(image)
            selected.append(image)
        elif random()< 0.5:
            #use the image but dont add it to the pool
            selected.append(image)
        else:
            #replace an exisitng image and use replaced image
            ix = randint(0,len(pool))
        selected.append(pool[ix])
        pool[ix]=image
        return asarray(selected)
