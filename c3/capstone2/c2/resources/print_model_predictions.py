from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

'''
Load model.

# ImageGenerator: use model_evaluate_val1

# image_dataset_from_directory: use model_evaluate_val2
    data_dir = path to folder containing fire & non_fire subfolders

Plots 6 images with original image label and whether the model predicted correct.

Plots ROC curve and displays confusion matrix for image_dataset_from_directory generator only.
'''

def model_evaluate_val1(ax, model, batch_size=1):
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

def model_evaluate_val2(ax, data_dir, model, batch_size=1):
    X_test = image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=4,
      shuffle=True,
      image_size=(256, 256),
      batch_size=batch_size)

    print(model.evaluate(X_test))

    images = []
    results = []
    labels = []

    for i, (image, label) in enumerate(X_test.take(6)):
      prediction = model.predict(image)

      if (prediction < 0.5) != label:
        result = 'Correct'
      else:
        result = 'Incorrect'

      results.append(result)
      labels.append(label)
      
      images.append(image[0].numpy().astype("uint8"))
      if i > 4: break
    print('Results stored.')
    labels = ['FIRE' if label==0 else 'So not fire' for label in labels]
    print('Labels updated.')    
    for i, (label, result, image) in enumerate(zip(labels, results, images)):
        ax[i//3, i%3].imshow(image)
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('{}, Predicted {}'.format(label, result))
    return ax

def plot_roc(ax, data_dir, model):

  X_test2 = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=16)

  print(model.evaluate(X_test2))

  y = np.concatenate([y for x, y in X_test2], axis=0)
  predictions = model.predict(X_test2, verbose=2)
  y_pred = predictions.ravel()
  y_pred = y_pred > 0.5
  fpr, tpr, thresholds = roc_curve(y, y_pred)
  auc_ = auc(fpr, tpr)
  print(confusion_matrix(y, y_pred))

  ax.plot([0, 1], [0, 1], 'k--')
  ax.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_))
  ax.set_xlabel('False positive rate')
  ax.set_ylabel('True positive rate')
  ax.set_title('ROC curve')

  return ax

if __name__ == '__main__':
    
    # Define your data directory for image_dataset_from_directory
    # This path should lead to one folder containing two classes' subfolders
    #data_dir = ''

    # Load your saved model '*.h5'
    model = load_model('model.h5')
    print('Model loaded.')

    # Plotting images and whether prediction was correct
    fig, ax = plt.subplots(2,3, figsize=(10,6))
    fig.subplots_adjust(wspace=.05)

    #model_evaluate_val1(ax, model)
    #model_evaluate_val2(ax, data_dir, model)

    fig.savefig('predictions.jpeg')

    # Displaying confusion matrix and roc curve
    fig, ax = plt.subplots(figsize=(10,10))

    ax = plot_roc(ax, data_dir, model)

    plt.show()
    plt.savefig('roccurve.jpeg')