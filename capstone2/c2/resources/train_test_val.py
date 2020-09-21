from PIL import Image
import numpy as np
import shutil
from shutil import copyfile
import os

fire_path = '../data/fire_images'
notfire_path = '../data/non_fire_images'

def load_data(path1=fire_path, path2=notfire_path):
    '''
    This file assumes you have downloaded the kaggle fire and not fire images into the data folder with subfolders:
        'fire_images'  and  'non_fire_images'
        And that you wish to use ImageGenerators & the flow from directory method to generate train, test, and validation images.
    OUTPUTS
    -------
        * none: creates new files in data folder: 'train', 'test', 'val'
                each with subfolders: 'fire_images' and 'non_fire_images'
    '''

    fire_path = os.listdir(path1)
    non_fire_path = os.listdir(path2)

    fire_images = os.listdir(fire_path)
    non_fire_images = os.listdir(non_fire_path)

    # shuffle
    np.random.shuffle(fire_images)
    np.random.shuffle(non_fire_images)
    # train folder
    for i, img1 in enumerate(fire_images):
        shutil.copyfile(img1, '../data/train/fire')
        if i > 500:
            break
    for i, img2 in enumerate(non_fire_images):
        shutil.copyfile(img2, '../data/train/notfire')
        if i > 500:
            break
    
    # shuffle
    np.random.shuffle(fire_images)
    np.random.shuffle(non_fire_images)
    # test folder
    for i, img01 in enumerate(fire_images):
        shutil.copyfile(img01, '../data/test/fire')
        if i > 200:
            break
    for i, img02 in enumerate(non_fire_images):
        shutil.copyfile(img02, '../data/test/notfire')
        if i > 200:
            break

    # shuffle
    np.random.shuffle(fire_images)
    np.random.shuffle(non_fire_images)
    # val folder
    for i, img001 in enumerate(fire_images):
        shutil.copyfile(img001, '../data/val/fire')
        if i > 200:
            break
    for i, img002 in enumerate(non_fire_images):
        shutil.copyfile(img002, '../data/val/notfire')
        if i > 200:
            break