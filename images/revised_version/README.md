Updated 8/30/2020

Images for new README to replace initial project submitted 8/28/2020.

**********************************************
# Forest Fire Detection
**********************************************

#### Author: Chelsea Zaloumis
#### Galvanize DSI Capstone 2
*Last update: 8/30/2020*

 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/cawildfire.jpeg" width="75%" height="75%"/>
 </p>

# Background & Motivation

Today (August 2020), 92 forest fires are burning on approximately 1.5 million acres in 13 states. There are currently more than 24,000 wildland firefighters assigned and distributed to tackle these incidents according to The National Interagency Fire Center. Forest fires are not only destructive and dangerous, but their impact lingers with unhealthy air quality. 

The past two weeks left many states in smoke (or worse...) and I was encouraged to begin exploration of fire detection and eventually fire prediction. My goal in this project is to create a convolutional neural network to accurately identify forest fires in images. Readers can follow along in the Google Colab notebook in the scr folder or look at the inidividual .py files organized amongst resources and src folders.

# Data

Kaggle fire images dataset: https://www.kaggle.com/phylake1337/fire-dataset
 * 2 folders
 * fireimages folder contains 755 outdoor-fire images some of them contains heavy smoke
 * non-fireimages folder contains 244 nature images (eg: forest, tree, grass, river, people, foggy forest, lake, animal, road, and waterfall)

# Image Preprocessing

The initial dataset is imbalanced. To remedy this, I have several options:

  * Web scrap/download additional nature images from the internet
  * Generate augmented images of the nature images already in the dataset
  * Undersample from the fire class of images 
  
I opted for generating augmented images. In the resources folder, readers can find 'gen_images.py' by which they can generate augmented images for either fire or non fire classes. The file assumes there is a path to one data folder containing both fire_images and non_fire_images subfolders and that the image formats are .png.

Examples of images comprising the dataset:
 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/test_image_examples.jpeg" />
 </p>

# CNN Model 1

I constructed a first convolutional neural network based off the keras blog post for "Building Powerful Image Classification Models Using Very Little Data." https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Code for this model can be found in resources as 'model1.py'. Note that this file uses a flow_from_directory image generator.

 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m1_summary.png" width="50%" height="50%"/>
 </p>
 
 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m1_loss_acc50epoch.jpeg" />
 </p>

This model has o.k. training and test accuracy and is doing slightly better than guessing for fire in images. See example images below with their corresponding label and if the model identified fire correctly. Evaluated on hold-out images results:

 * Loss: 0.63
 * Accuracy: 0.69

 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m1_predictions50epoch.jpeg" />
 </p>

This model's results become less reliable when we look at the ROC curve and corresponding confusion matrix:
```
[[ 59  93]
 [ 58 112]]
```
 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m1_roccurve.jpeg" width="50%" height="50%" />
 </p>

# CNN Model 2

The second convolutional neural network references a Tensorflow classification tutorial. https://www.tensorflow.org/tutorials/images/classification

Code for this model can be found in resources as 'model2.py'. This model was defined with a best-weights checkpoint, model save checkpoint, and early stop checkpoint. Note that this file uses a image_dataset_from_directory image generator. This same method is used in the Google Colab notebook.

 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m2_summary.png" width="50%" height="50%" />
 </p>
 
 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m2_loss_acc_es.jpeg" />
 </p>

This model performs much better than the first CNN model. Evaluated on hold-out images:

 * Loss: 0.19
 * Accuracy: 0.92
 
 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m2_predictions_es.jpeg" />
 </p>
 
However this model is not tuned for the default threshold = 0.5 as we can see from the confusion matrix and ROC curve.

```
[[ 73  79]
 [ 65 100]]
```
 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m2_roccurve_es.png" width="50%" height="50%" />
 </p>

# Optimal Threshold Tuning

In order to find the optimal threshold for Model 2, I first predicted class labels and then evaluated them using the F1 Score, which is the harmonic mean of precision and recall. 

