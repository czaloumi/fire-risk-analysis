import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

<<<<<<< HEAD
'''
Second saved model. Checkpoint (defined in if name == main) compares specified score each epoch.

Saves weights of best score.
'''

=======
>>>>>>> dcfc837570fb3491deeadd9e7da8e60f5adc06ae
def model2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 3), activation = 'relu'))
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
    model.add(Dense(units = 1, activation = 'sigmoid'))
    
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
    
    checkpoint = ModelCheckpoint(
            filepath='best_weights.hdf5',
            monitor = 'val_accuracy',
            verbose=1,
            save_best_only=True)
    
    model = model2()
    
    history = model.fit(
            train_generator,
            steps_per_epoch=10,
            epochs=10,
            callbacks=[checkpoint],
            validation_data=validation_generator,
            validation_steps=10)
    
    model.summary()

    fig, ax = plt.subplots(1, 2, figsize=(10,6))

    ax[0].plot(history.history['loss'], label='train', linestyle='-.', color='orange')
    ax[0].plot(history.history['val_loss'], label='test', linestyle='--', color='green')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[1].plot(history.history['accuracy'], label='train', linestyle='-.', color='orange')
    ax[1].plot(history.history['val_accuracy'], label='test',linestyle='--', color='green')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    fig.legend()
    fig.show() 
    fig.savefig('../images/m2.jpeg')