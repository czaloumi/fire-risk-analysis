import os
import numpy as np
from numpy import random
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

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
    Reads in image, coverts to array, reshapes, applies datagen 4 times.
    Repeats num times.

    INPUT
    -----
    path: string; path to data folder from gen_images.py location
    num: int; second to last image to augment
    fire: boolean; if True, augment/generate fire images; if False, augment/generate non_fire images
    '''
    
    if fire:
        for i in tqdm(range(1, num+1)):
            
            img = Image.open(f'{path}/fire.{i}.png')
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i=0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=f'{path}/', save_prefix='fire', save_format='png'):
                i += 1
                if i > 20:
                    break
    else:
        for i in tqdm(range(1, num+1)):
            img = Image.open(f'{path}/non_fire.{i}.png')
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i=0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=f'{path}/', save_prefix='non_fire', save_format='png'):
                i += 1
                if i > 4:
                    break