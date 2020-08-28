from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import PIL
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow as tf
import pathlib

data_dir = '../data/fire_dataset/'
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 256
img_width = 256
epochs=50

X_train = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

X_test = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

for image_batch, labels_batch in X_train:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

'''
class_names = X_train.class_names
print(class_names)

plt.figure(figsize=(10,10))
for images, labels in X_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()
plt.savefig('../images/fire_notfire_mix.png')
'''

# use buffered prefetching so we can yield data from disk without having I/O become blocking. 

AUTOTUNE = tf.data.experimental.AUTOTUNE

X_train = X_train.cache().shuffle(32).prefetch(buffer_size=AUTOTUNE)
X_test = X_test.cache().shuffle(32).prefetch(buffer_size=AUTOTUNE)


'''
for image_batch, labels_batch in X_train.take(1):
    print(image_batch.shape)
    print(labels_batch.shape)
    break
'''

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(1, activation='sigmoid')
])
print('Model loaded.')

model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
print('Model compiled.')

history = model.fit(
            X_train,
            steps_per_epoch=10,
            epochs=epochs,
            validation_data=X_test)
print('Model fitted.')

from cnn_fire import plot_loss_val

fig, ax = plt.subplots(1,2, figsize=(10,6))
plot_loss_val(ax, history)
fig.legend()
fig.savefig('../images/model_loss_acc_1.jpeg')