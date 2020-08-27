import numpy as np
import os
import shutil
import random

fire_path = '../data/fire_dataset/fire_images/'
path1 = '../data/fire_dataset/test/fire_images'
path2 = '../data/fire_dataset/val/fire_images'

test_fire = os.listdir(path1)
val_fire = os.listdir(path2)

'''
for img in test_fire:
    os.remove(img)

for img in val_fire:
    os.remove('../data/fire_dataset/val/fire_images/{}'.format(img))
'''

# Now that I've removed all of the 'non fire' images in the test and validation folder's 'fire' folders, i'll add 229 in each (amount that I deleted)

num = 229
fire_images = os.listdir(fire_path)

'''
for i in range(num):
    fire_images = random.sample(fire_images, len(fire_images))
    shutil.copy('../data/fire_dataset/fire_images/{}'.format(fire_images[i]), path1)
    fire_images = random.sample(fire_images, len(fire_images))
    shutil.copy('../data/fire_dataset/fire_images/{}'.format(fire_images[i]), path2)
'''

test_fire = os.listdir(path1)
val_fire = os.listdir(path2)