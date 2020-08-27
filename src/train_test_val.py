from PIL import Image
import numpy as np
from shutil import copyfile
import os

'''
This file assumes you have downloaded the kaggle fire and not fire images into the data folder with subfolders:
    'fire_images'  and  'non_fire_images'

OUTPUTS
-------
    * none: creates new files in data: 'train', 'test', 'val'
'''

fire_path = '../data/fire_images'
notfire_path = '../data/non_fire_images'

def loadImages(path1=fire_path, path2=notfire_path):
    fire_images = listdir(path1)
    non_fire_images = listdir(path2)

    # shuffle
    np.random.shuffle(fire_images)
    np.random.shuffle(non_fire_images)
    # train folder
    for i, img1 in enumerate(fire_images):
        shutil.copyfile(os.path.join(f'{}/{}'.format(path1, img1), '../data/train/fire'))
        if i > 500:
            break
    for i, img2 in enumerate(non_fire_images):
        shutil.copyfile(os.path.join(f'{}/{}'.format(path2, img2), '../data/train/notfire'))
        if i > 500:
            break
    
    # shuffle
    np.random.shuffle(fire_images)
    np.random.shuffle(non_fire_images)
    # test folder
    for i, img01 in enumerate(fire_images):
        shutil.copyfile(os.path.join(f'{}/{}'.format(path1, img01), '../data/test/fire'))
        if i > 200:
            break
    for i, img02 in enumerate(non_fire_images):
        shutil.copyfile(os.path.join(f'{}/{}'.format(path2, img02), '../data/test/notfire'))
        if i > 200:
            break

    # shuffle
    np.random.shuffle(fire_images)
    np.random.shuffle(non_fire_images)
    # val folder
    for i, img001 in enumerate(fire_images):
        shutil.copyfile(os.path.join(f'{}/{}'.format(path1, img001), '../data/val/fire'))
        if i > 200:
            break
    for i, img002 in enumerate(non_fire_images):
        shutil.copyfile(os.path.join(f'{}/{}'.format(path2, img002), '../data/val/notfire'))
        if i > 200:
            break