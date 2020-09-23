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

'''
This notebook was made to overcome this message that eventually began popping up in 'cnn_fire.py':
  Filling up shuffle buffer (this may take a while): 107 of 128
  Shuffle buffer filled.
  * this would run for about 45 minutes to get through 10 epochs

I then began receiving this error:
  The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
  * still am unsure why i was getting this error - a place for more research
  * did not have shuffle buffer or this error in Colab!

Moved to Google Colab to remedy.
'''

data_dir = '../data/fire_dataset0/'
data_dir = pathlib.Path(data_dir)

batch_size = 16
img_height = 256
img_width = 256
epochs=10

X_train = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  shuffle=True,
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
    print(image_batch[0].shape)
    print(image_batch.shape)
    print(labels_batch.shape)
    break

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

# use buffered prefetching so we can yield data from disk without having I/O become blocking. 
AUTOTUNE = tf.data.experimental.AUTOTUNE
#The buffer_size in Dataset.shuffle() can affect the randomness of your dataset, and hence the order in which elements are produced.
#The buffer_size in Dataset.prefetch() only affects the time it takes to produce the next element.
X_train = X_train.cache().repeat().shuffle(16).prefetch(buffer_size=AUTOTUNE)
X_test = X_test.cache().repeat().prefetch(buffer_size=AUTOTUNE)


model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.Conv2D(16, 3, activation='relu'),
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
            validation_data=X_test,
            validation_steps=5)
print('Model fitted.')

from cnn_fire import plot_loss_val

fig, ax = plt.subplots(1,2, figsize=(10,6))
plot_loss_val(ax, history)
fig.legend()
fig.savefig('../images/model_loss_acc_1.jpeg')