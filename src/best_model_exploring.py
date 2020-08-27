from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
Model from cnn_model2.py -> saved in best_model.py

Iterates through batch size of 1 in validation data and pulls the image, resulting detection, and original label for plotting.

Plots 6 images.
'''

model = load_model('bestmodel.h5')

def model_evaluate_val(ax, model, batch_size=1):
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
            '../data/val',
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True)

    print(model.evaluate(generator))

    images = []
    results = []
    labels = []
    for i, (image, label) in enumerate(generator):
        prediction = model.predict(image)
        if (prediction < .5) != (label):
            result = 'Correct'
        else:
            result = 'Incorrect'
        
        results.append(result)
        labels.append(label)

        im = image.reshape(256,256,3) 
        im = (im*255).astype(np.uint8) 
        ims = Image.fromarray(im)
        images.append(ims)

        if i > 4:
            break

    labels = ['FIRE' if label==0 else 'So not fire' for label in labels]

    for i, (label, result, image) in enumerate(zip(labels, results, images)):
        ax[i//3, i%3].imshow(image)
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('{}, Predicted {}'.format(label, result))
    return ax

fig, ax = plt.subplots(2,3, figsize=(10,6))
fig.subplots_adjust(wspace=.05)
model_evaluate_val(ax, model)
fig.savefig('../images/test1.jpeg')
print(model.evaluate(holdout_generator))

images = []
results = []
labels = []
for i, (image, label) in enumerate(holdout_generator):
    prediction = model.predict(image)
    if (prediction < .5) != (label):
        result = 'Correct'
    else:
        result = 'Incorrect'
    
    i_s.append(i)
    results.append(result)
    labels.append(label)

    im = image.reshape(256,256,3) 
    im = (im*255).astype(np.uint8) 

    ims = Image.fromarray(im)
    #ims.save('{}_{}_{}.png'.format(i, label, prediction))
    images.append(ims)

    if i > 4:
        break

labels = ['FIRE' if label==0 else 'So not fire' for label in labels]

fig, ax = plt.subplots(2,3, figsize=(10,6))
fig.subplots_adjust(wspace=.05)
for i, (label, result, image) in enumerate(zip(labels, results, images)):
    ax[i//3, i%3].imshow(image)
    ax[i//3, i%3].axis('off')
    ax[i//3, i%3].set_title('{}, Predicted {}'.format(label, result))
fig.savefig('../images/m2testonholdout.jpeg')
plt.show()
