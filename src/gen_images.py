import os
import numpy as np
from numpy import random
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

'''
The gen_images function was initially used to generate augmented images using the ImageDataGenerator defined below.

I had planned to convert all these images to arrays, append the 'fire' and 'nonfire' arrays to one giant matrix X.
Similarly, I was creating a 'y' labels array based on the folder the images came out of.

This was crashing my kernel and I figured it was too much for my computer's memory.

I then transitioned to the blog's (https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
method of generating augmented images on the fly in the the model's .fit

I used gen_images in main.ipynb to generate about ~800-1,200 augmented images for each class.

Code in main.ipynb also reflects my manual 'im_to_array' function and the attempt to save the arrays that was crashing my kernel.
'''

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def gen_images(path, num, fire):
    '''
    Reads in image, coverts to array, reshapes, applies datagen 20 times.
    Repeats num times.
    '''
    if fire:
        for i in tqdm(range(28, num)):
            
            img = load_img('{}/fire.{}.png'.format(path, i))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i=0
            for batch in datagen.flow(x, batch_size=1, save_to_dir='{}/'.format(path), save_prefix='fire', save_format='png'):
                i += 1
                if i > 20:
                    break
    else:
        for i in tqdm(range(28, num)):
            img = load_img('{}/non_fire.{}.png'.format(path, i))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i=0
            for batch in datagen.flow(x, batch_size=1, save_to_dir='{}/'.format(path), save_prefix='non_fire', save_format='png'):
                i += 1
                if i > 20:
                    break