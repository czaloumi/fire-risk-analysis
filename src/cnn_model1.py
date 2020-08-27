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

<<<<<<< HEAD
'''
My first model following the keras blog post. 
Experimented with changing my last activation layer from sigmoid <-> softmax.

Model did not perform well -> couldn't understand why until I found non fire images in my train/test/val fire folders. Whoops!
'''

=======
>>>>>>> dcfc837570fb3491deeadd9e7da8e60f5adc06ae
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
<<<<<<< HEAD
    model.add(Activation('sigmoid'))
=======
    model.add(Activation('softmax'))
>>>>>>> dcfc837570fb3491deeadd9e7da8e60f5adc06ae
    
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

    # 1st run: dropout .5 and last activation function sigmoid
        # loss: 0.1194 - accuracy: 0.9375 - val_loss: 3.2206 - val_accuracy: 0.5375
    # 2nd run: dropout .8 and last activation function softmax
        # loss: 8.0058 - accuracy: 0.4750 - val_loss: 7.0528 - val_accuracy: 0.5375
    # 3rd run: dropout .7 and last activation function sigmoid
        # loss: 0.0218 - accuracy: 1.0000 - val_loss: 3.6720 - val_accuracy: 0.5500
    
    # Last run (Wed 8/26): one last Dense layer, w/sigmoid activation; loss='binary_crossentropy'
    
    model = model1()
    
    history = model.fit(
            train_generator,
            steps_per_epoch=10,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=10)

    # Ran this ~4-5 times => naming the weights each time - only kept several
    model.save_weights('m1_lastlast.h5')

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
    fig.savefig('../images/lastlast_model1.jpeg')
    