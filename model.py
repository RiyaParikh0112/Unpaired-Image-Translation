import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model

# define input shape of the model
image_shape = (256, 256, 3)

# load models
g_model_AtoB = load_model('g_model_AtoB.h5')
g_model_BtoA = load_model('g_model_BtoA.h5')

# define a function to perform image translation
def translate_image(input_image, direction):
    # load the input image
    img = load_img(input_image, target_size=image_shape)
    # convert the image to a numpy array
    img = img_to_array(img)
    # scale the pixel values to the range of [-1, 1]
    img = (img - 127.5) / 127.5
    # expand the dimensions of the image to match the model's input shape
    img = np.expand_dims(img, axis=0)
    # define the model to be used for translation
    if direction == 'AtoB':
        model = g_model_AtoB
    else:
        model = g_model_BtoA
    # perform the translation
    translated_image = model.predict(img)
    # rescale the pixel values to the range of [0, 255]
    translated_image = (translated_image + 1) / 2.0 * 255.0
    # convert the numpy array to an image object
    translated_image = translated_image[0].astype(np.uint8)
    # return the translated image
    return translated_image

# create a user interface to allow users to upload images and visualize the translated images
inputs = [
    gr.inputs.Image(label='Input Image'),
    gr.inputs.Radio(['AtoB', 'BtoA'], label='Translation Direction')
]

outputs = gr.outputs.Image(label='Translated Image')

title = 'CycleGAN Image Translation'
description = 'Translate an image from one domain to another using CycleGAN.'

gr.Interface(fn=translate_image, inputs=inputs, outputs=outputs, title=title, description=description).launch()
