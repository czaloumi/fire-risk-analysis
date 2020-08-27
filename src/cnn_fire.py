'''
Train a deep CNN to identify fire in images.

96% validation accuracy in 10 epochs.
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
from shutil import copyfile
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random

fire_path = '../data/fire_images'
notfire_path = '../data/non_fire_images'

def load_data(path1=fire_path, path2=notfire_path):
    '''
    This file assumes you have downloaded the kaggle fire and not fire images into the data folder with subfolders:
        'fire_images'  and  'non_fire_images'

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

def model(activation='sigmoid', num_classes=1, input_shape=(256, 256, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) 
    # second convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) # antes era 0.25
    # third convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) # antes era 0.25
    # flatten
    model.add(Flatten())
    # full connection
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(units = num_classes, activation = activation))
    print('- - - - -> Model defined.')
    return model

def compile_model(model, metrics, optimizer='adam'):
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=metrics)
    print('- - - - - - -> Model compiled.')
    return model

def train_model(model, batch_size, epochs):

    print('- - - - - - - - -> Using data augmentation.')

    # Training data generator
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # Test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Training data
    train_generator = train_datagen.flow_from_directory(
            '../data/train',  # target directory
            target_size=(256, 256),  # resize images to 256 x 256
            batch_size=batch_size,
            class_mode='binary', # since we use binary_crossentropy loss, we need binary label,
            shuffle=True)  # same shuffle each time

    # Validation data
    validation_generator = test_datagen.flow_from_directory(
            '../data/val',
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='binary')
    
    # Checkpoints to identify best weights
    checkpoint = ModelCheckpoint(
            filepath='best_weights.hdf5',
            monitor = 'val_accuracy',
            verbose=1,
            save_best_only=True)
    
    model.fit(
            train_generator,
            steps_per_epoch=10,
            epochs=epochs,
            callbacks=[checkpoint],
            validation_data=validation_generator,
            validation_steps=10)

    return model

def save_model(model, model_title):
    model.save(model_title)
    print('- - - - - - -> Model saved as {}'.format(model_title))

def eval(model):
    test_datagen = ImageDataGenerator(rescale=1./255)

    holdout_generator = test_datagen.flow_from_directory(
            '../data/test',
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True)

    print('- - - - -> Evaluating on hold-out images...')
    l, a = model.evaluate(holdout_generator)
    print('- - - - -> Loss: {}, Accuracy: {}'.format(l, a))

def plot_loss_val(ax, model):
    '''
    Plots model's train & test loss per epoch and model's train & test accuracy per epoch.
    '''
    ax[0].plot(model.model['loss'], label='train', linestyle='-.', color='orange')
    ax[0].plot(model.model['val_loss'], label='test', linestyle='--', color='green')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[1].plot(model.model['accuracy'], label='train', linestyle='-.', color='orange')
    ax[1].plot(model.model['val_accuracy'], label='test',linestyle='--', color='green')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')

    return ax

def show_me_the_predictions(ax, model):
    '''
    Iterates through batch size of 1 in validation data and pulls the image, resulting detection, and original label for plotting.

    Plots 6 images.
    '''
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
            '../data/val',
            target_size=(256, 256),
            batch_size=1,
            class_mode='binary',
            shuffle=True)

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


if __name__ == "__main__":
    batch_size = 16
    num_classes = 1
    epochs = 10
    activation = 'sigmoid'
    optimizer = 'adam'
    metrics = ['accuracy']
    
    model = model(activation, num_classes)

    model = compile_model(model, metrics, optimizer)
    
    # load model weights                                        # remove later
    model.load_weights('best_weights.hdf5')

    model = train_model(model, batch_size, epochs)

    print(model.summary())

    #save_model(model, 'fire_trained_model.h5')

    eval(model)

    fig, ax = plt.subplots(1,2, figsize=(10,6))
    plot_loss_val(ax, model)
    fig.legend()
    fig.savefig('../images/model_loss_acc.jpeg')

    fig, ax = plt.subplots(2,3, figsize=(10,6))
    fig.subplots_adjust(wspace=.05)
    show_me_the_predictions(ax, model)
    fig.savefig('../images/heres_the_predictions.jpeg')