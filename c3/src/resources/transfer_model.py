from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import PIL
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
import pathlib
from tqdm import tqdm

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import warnings
warnings.filterwarnings("ignore")
from itertools import chain

#################################################################################


#################################################################################
# GENERATORS
#################################################################################

X_train = image_dataset_from_directory(
  directory,
  validation_split=0.2,
  subset="training",
  seed=123,
  shuffle=True,
  image_size=(img_height, img_width),
  batch_size=batch_size)

X_test = image_dataset_from_directory(
  directory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

X_test2 = image_dataset_from_directory(
    directory,
    validation_split=0.2,
    subset='validation',
    seed=12345,
    image_size=(img_height, img_width),
    batch_size=1)

AUTOTUNE = tf.data.experimental.AUTOTUNE
X_train = X_train.cache().repeat().shuffle(16).prefetch(buffer_size=AUTOTUNE)
X_test = X_test.cache().repeat().prefetch(buffer_size=AUTOTUNE)

#################################################################################
# MODELS
#################################################################################

def create_model(input_size):
    
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=input_size),
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=input_size),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
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
'''
def add_model_head(base_model, n_categories):
    """
    Takes a base model and adds a pooling and a softmax output based on the number of categories
    Args:
        base_model (keras Sequential model): model to attach head to
        n_categories (int): number of classification categories
    Returns:
        keras Sequential model: model with new head
        """

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_categories, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_transfer_model(input_size, n_categories, weights = 'imagenet', model=Xception):
    """
    Creates model without top and attaches new head to it
    Args:
        input_size (tuple(int, int, int)): 3-dimensional size of input to model
        n_categories (int): number of classification categories
        weights (str or arg): weights to use for model
        model (keras Sequential model): model to use for transfer
    Returns:
        keras Sequential model: model with new head
        """
    base_model = model(weights=weights,
                      include_top=False,
                      input_shape=input_size)
    model = add_model_head(base_model, n_categories)
    return model
'''
def transfer_model(input_size, weights = 'imagenet'):
        # note that the "top" is not included in the weights below
        base_model = Xception(weights=weights,
                          include_top=False,
                          input_shape=input_size)
        
        model = base_model.output
        
        # add new head
        model = GlobalAveragePooling2D()(model)
        predictions = Dense(1, activation='sigmoid', name='visualized_layer')(model)
        model = Model(inputs=base_model.input, outputs=predictions)

        return model

def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

#################################################################################
# EVALUATE MODELS
#################################################################################

def model_evaluate_val(ax, model, generator):

    for image, label in generator.take(1):
        prediction = model.predict(image)

        if (prediction < 0.5) != label:
            result = 'Correct'
        else:
            result = 'Incorrect'

        img = image[0].numpy().astype("uint8")
    
        label = 'Fog' if label==0 else 'Smoke'

    ax.imshow(img)
    ax.axis('off')
    ax.set_title('{}, Predicted {}'.format(label, result))
    return ax

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: array, array, array

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    # tpr = tp / tp+fn
    # fpr = fp / fp+tn
    
    df = pd.DataFrame({'probabilities': probabilities, 'y': labels})
    df.sort_values('probabilities', inplace=True)

    actual_p = df.y.sum()
    actual_n = df.shape[0] - df.y.sum()
    
    df['tn'] = (df.y == 0).cumsum()
    df['fn'] = df.y.cumsum()
    df['fp'] = actual_n - df.tn
    df['tp'] = actual_p - df.fn
    
    df['fpr'] = df.fp/(df.fp + df.tn)
    df['tpr'] = df.tp/(df.tp + df.fn)
    df['precision'] = df.tp/(df.tp + df.fp)
    df['F1'] = 2*((df.tp/(df.tp + df.fp)) * (df.tp/(df.tp + df.fn)))/((df.tp/(df.tp + df.fp)) + (df.tp/(df.tp + df.fn)))
    df = df.reset_index(drop=True)
    return df

def plot_roc(ax, df):
    auc_ = round(auc(df.fpr, df.tpr),2)
    ax.plot([1]+list(df.fpr), [1]+list(df.tpr), label=f'AUC = {auc_}', color='r')
    ax.plot([0,1],[0,1], 'k', label="random")
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()


if __name__ == "__main__":

    directory = '../data/cleaned_data'

    batch_size = 16
    img_height = 256
    img_width = 256
    epochs=50

    model = load_model('transfer_model.h5')
    def print_model_properties(model, indices = 0):
    for i, layer in enumerate(model.layers[indices:]):
        print("Layer {} | Name: {} | Trainable: {}".format(i+indices, layer.name, layer.trainable))
    
    # PRINT ONE PREDICTION
    '''
    fig, ax = plt.subplots(figsize=(10,10))
    model_evaluate_val(ax, model, X_test2)
    fig.savefig('model_prediction_6.jpeg')
    '''
    # ROC CURVE
    '''
    fig, ax = plt.subplots(figsize=(10,10))
    df = roc_curve(probabilities, y)
    plot_roc(ax, df)
    fig.show()
    fig.savefig('roccurve.jpeg')
    '''

    i = 100
    while i > 0:
    change_trainable_layers(model, i) 
    print_model_properties(model)
    
    new_model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'mse'])
    print('- - -> Transfer Model compiled.')

    mc = ModelCheckpoint('transfer_model.h5', save_best_only=True)
    print('- - - - -> Checkpoints defined.')

    history = model.fit(
                X_train,
                steps_per_epoch=int(np.floor(1488/16)),
                epochs=10,
                callbacks=[mc, tensorboard],
                validation_data=X_test,
                validation_steps=np.floor(331/16))
    print('- - -> Transfer Model trained.')

    print(model.summary())

    model.save('transfer_model_just_in_case.h5')
    i-=4

    '''
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].plot(history.history['loss'], label='train', linestyle='-.', color='pink')
    ax[0].plot(history.history['val_loss'], label='test', linestyle='--', color='purple')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[1].plot(history.history['accuracy'], label='train', linestyle='-.', color='pink')
    ax[1].plot(history.history['val_accuracy'], label='test',linestyle='--', color='purple')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    fig.legend()
    fig.savefig('transfermodel_loss_acc.jpeg')
    '''

    file_path_to_trans_model = 'transfer_model_just_in_case.h5'
    best_trans_model = load_model(file_path_to_trans_model)

    y = np.concatenate([y for x, y in X_test], axis=0)
    probs = best_trans_model.predict(X_test)
    probs = probs.reshape(1,-1)
    probabilities = list(chain.from_iterable(probs))
    # ROC CURVE
    fig, ax = plt.subplots(figsize=(10,10))
    df = roc_curve(probabilities, y)
    plot_roc(ax, df)
    fig.show()
    fig.savefig('transfermodel_roccurve.jpeg')