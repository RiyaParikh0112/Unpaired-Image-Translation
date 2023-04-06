#loading the dataset
import os
from os import listdir
import numpy as np
from numpy import asarray
from numpy import vstack
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
#load and plot the prepared dataset
from numpy import load
from matplotlib import pyplot

#the photos are of sqaure size 256*256
#we create an array of images for Category A and another for Category B. Both the arrays are then saved to a new file in compressed Numpy array format.

#loading all the images in a directory into memory.
def load_images(path, size=(256,256)):
    data_list = list()
    #list all the names of the files within the directory(enumerate)
    for filename in listdir(path):
        #load and resize the image
        pixels = load_img(path+filename, target_size=size)
        #now convert to numpy array
        pixels = img_to_array(pixels)
        #store to memory
        data_list.append(pixels)
        return asarray(data_list)
path = '/kaggle/input/cyclegan/vangogh2photo/vangogh2photo/'
#load Dataset A
dataA1 = load_images(path + 'trainA/')
dataA2 = load_images(path +'testA/')
dataA = vstack((dataA1,dataA2))
print('Dataset A loaded', dataA.shape)
#load Dataset B
dataB1 = load_images(path + 'trainB/')
dataB2 = load_images(path +'testB/')
dataB = vstack((dataB1,dataB2))
print('Dataset A loaded', dataB.shape)
#save as compressed numpy array
filename = 'vangogh2photo_comp.npz'
savez_compressed(filename, dataA,dataB)
print('Saved Dataset:', filename)
#the size is larger than the raw images as we are storing pixel values as 32-bit floating point values.

data = load('vangogh2photo_comp.npz')
dataA, dataB = data['arr_0'], data['arr_1']
print('Loaded: ', dataA.shape, dataB.shape)
# plot source images
n_samples = 2
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(dataA[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(dataB[i].astype('uint8'))
pyplot.show()
