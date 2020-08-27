import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
NOTES
-----
        My first model following the keras blog post. 
        Experimented with changing my last activation layer from sigmoid <-> softmax.
        Experimented with adding HIGH dropouts = 0.8 in the last two dense layers to copensate for overfitting.

        Model did not perform well -> couldn't understand why until I found non fire images in my train/test/val fire folders. Whoops!

OUTPUTS 
-------
        * saves model weights to 'm1_weights.h5'
        * prints model summary
        * prints model evaluation loss and validation accuracy on hold-out images
        * plots train vs. test loss and accuracy
        * saves train vs. test loss and accuracy plot as 'm1_loss_acc.jpeg'
'''

def model1():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(256, 256, 3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())  # converts 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1))    # = same # of nodes for classification - each node has a probability associated with it using softmax
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    return model

if __name__ == "__main__":

    batch_size = 16

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            '../data/fire_dataset/train',  # target directory
            target_size=(256, 256),  # resize images to 256 x 256
            batch_size=batch_size,
            class_mode='binary', # since we use binary_crossentropy loss, we need binary label,
            shuffle=True)  # same shuffle each time

    validation_generator = test_datagen.flow_from_directory(
            '../data/fire_dataset/val',
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True)
    
    holdout_generator = test_datagen.flow_from_directory(
            '../data/fire_dataset/test',
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True)
    
    model = model1()
    
    history = model.fit(
            train_generator,
            steps_per_epoch=10,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=10)

    model.save_weights('m1_weights.h5')

    model.summary()
    
    model.evaluate(holdout_generator)

    fig, ax = plt.subplots(1, 2, figsize=(10,6))

    ax[0].plot(history.history['loss'], label='train', linestyle='-.', color='red')
    ax[0].plot(history.history['val_loss'], label='test', linestyle='--', color='blue')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[1].plot(history.history['accuracy'], label='train', linestyle='-.', color='pink')
    ax[1].plot(history.history['val_accuracy'], label='test',linestyle='--', color='blue')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    fig.legend()
    fig.show() 
    fig.savefig('../images/m1_loss_acc.jpeg')
    