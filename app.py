import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
from PIL import Image

def load_image(img):
    with open(img.name, 'rb') as f:
        img_data = f.read()
    return img_data

def translate_image(image):
    # load the model
    cust = {'InstanceNormalization': InstanceNormalization}
    model_AtoB = load_model('/Users/riyaparikh/Desktop/ML Projects /CycleGAN/g_model_BtoA_004000.h5', cust)

    # convert to numpy array
    image_np = np.array(image)
#     image = Resizing(256, 256)(image)
    # normalize pixels
    image_np = (image_np - 127.5) / 127.5
    # add batch dimension
    image_np = np.expand_dims(image_np, axis=0)
    # translate image
    image_tar = model_AtoB.predict(image_np)
    # scale from [-1,1] to [0,1]
    image_tar = (image_tar + 1) / 2.0
    # convert back to PIL image
    image_tar = (image_tar[0] * 255).astype(np.uint8)
    image_tar = Image.fromarray(image_tar)
    return image_tar



# create the interface
input_image = gr.inputs.Image(type="pil")
output_image = gr.outputs.Image(type="pil")

iface = gr.Interface(fn=translate_image, inputs=input_image, outputs=output_image,
                     title="Van Gogh-ify Your Images🎨👨‍🎨",
                     description="Translate an image from one domain to another using CycleGAN")

iface.launch()
