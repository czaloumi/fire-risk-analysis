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

'''
Model from cnn_model2.py

Saves model
'''

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
    # load best weights
model.load_weights('best_weights.hdf5')
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # save this model
model.save('bestmodel.h5')