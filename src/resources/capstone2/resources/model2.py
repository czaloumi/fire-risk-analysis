'''
Model 2 following image classification tutorial at
https://www.tensorflow.org/tutorials/images/classification
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import ImageFile
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
from shutil import copyfile
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random

from print_model_predictions import *

# CNN Model 2 with 3x layers of 0.5 Dropout
# Note: if you run this cell more than once, you will want to rename:
# weights originally named 'm2_bestweights.hdf5'
# model originally named 'm2.h5'
# loss & accuracy plot originally named: 'm2_loss_acc.jpeg'

def model2():
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
    layers.Dense(1, activation='sigmoid', name='visualized_layer')
  ])
  return model

def compile_model(model, metrics, optimizer='adam'):
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=metrics)
    print('- - - - - - -> Model compiled.')
    return model

def train_model(model, data_dir, batch_size, epochs, img_height=256, img_width=256):

        print('- - - - - - - - -> Training model.')

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
        
        # Checkpoints to identify best weights
        checkpoint = ModelCheckpoint(
                filepath='m2_best_weights.hdf5',
                monitor = 'val_accuracy',
                verbose=1,
                save_best_only=True)
        print('Weights Checkpoint Defined.')

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint('m2.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        print('Early stopping and model checkpoint defined.')

        history = model.fit(
                train_generator,
                steps_per_epoch=10,
                epochs=epochs,
                callbacks=[checkpoint, es, mc],
                validation_data=validation_generator,
                validation_steps=10)
        print('Model Trained.')

        return model, history

def save_model(model, model_title):
    model.save(model_title)
    print('- - - - - - -> Model saved as {}'.format(model_title))

def eval(model, batch_size, img_height=256, img_width=256):
    X_test = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=4,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print('- - - - -> Evaluating on hold-out images...')
    l, a = model.evaluate(X_test)
    print('- - - - -> Loss: {}, Accuracy: {}'.format(l, a))

def plot_loss_val(ax, history):
    '''
    Plots model's train & test loss per epoch and model's train & test accuracy per epoch.
    '''
    ax[0].plot(history.history['loss'], label='train', linestyle='-.', color='orange')
    ax[0].plot(history.history['val_loss'], label='test', linestyle='--', color='green')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[1].plot(history.history['accuracy'], label='train', linestyle='-.', color='orange')
    ax[1].plot(history.history['val_accuracy'], label='test',linestyle='--', color='green')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')

    return ax

def plot_roc(ax, model, threshold=0.5):

  X_test2 = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=16)

  #print(model.evaluate(X_test2))

  y = np.concatenate([y for x, y in X_test2], axis=0)
  predictions = model.predict(X_test2, verbose=2)
  y_pred = predictions.ravel()
  y_pred = y_pred > threshold
  fpr, tpr, thresholds = roc_curve(y, y_pred)
  auc_ = auc(fpr, tpr)
  print(confusion_matrix(y, y_pred))

  ax.plot([0, 1], [0, 1], 'k--')
  ax.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_))
  ax.set_xlabel('False positive rate')
  ax.set_ylabel('True positive rate')
  ax.set_title('ROC curve')

  return ax

def labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

def tune_threshold(model):
  X_test2 = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=16)

  print(model.evaluate(X_test2))

  y = np.concatenate([y for x, y in X_test2], axis=0)
  yhat = model.predict(X_test2, verbose=2)

  thresholds = np.arange(0, 1, 0.001)
  
  scores = [f1_score(y, labels(yhat, t)) for t in thresholds]
    
  ix = np.argmax(scores)
  print(f'Threshold={thresholds[ix]}, F-Score={scores[ix]}')
  
  return thresholds[ix], scores[ix]

if __name__ == '__main__':
        # Define your data directory for image_dataset_from_directory
        # This path should lead to one folder containing two classes' subfolders
        #data_dir = ''

        img_height = 256
        img_width = 256
        batch_size = 16
        num_classes = 1
        epochs = 10
        activation = 'sigmoid'
        optimizer = 'adam'
        metrics = ['accuracy']

        model = model2()
        print('Model loaded.')

        model = compile_model(model, metrics, optimizer=)

        model, history = train_model(model, data_dir, batch_size, epochs)

        print(model.summary())

        save_model(model, '../images/m2.h5')

        eval(model, batch_size)

        fig, ax = plt.subplots(1,2, figsize=(10,6))
        plot_loss_val(ax, model)
        fig.legend()
        fig.savefig('../images/m2_loss_acc.jpeg')

        fig, ax = plt.subplots(2,3, figsize=(10,6))
        fig.subplots_adjust(wspace=.05)
        model_evaluate_val2(ax, data_dir, model, batch_size=1)
        fig.savefig('../images/m2_heres_the_predictions.jpeg')  

        # Displaying confusion matrix and roc curve
        fig, ax = plt.subplots(figsize=(10,10))

        ax = plot_roc(ax, data_dir, model)

        fig.show()
        fig.savefig('m2_roccurve.jpeg')    

        # Tune optimal threshold
        threshold, score = tune_threshold(model)

        fig, ax = plt.subplots(figsize=(10,10))
        
        ax = plot_roc(ax, model, threshold=threshold)
        
        fig.show()
        fig.savefig('m2_roccurve_optimalthreshold.jpeg')