**********************************************
# Forest Fire Detection
**********************************************

#### Author: Chelsea Zaloumis
#### Galvanize DSI Capstone 2
*Last update: 8/30/2020*

 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/cawildfire.jpeg" width="75%" height="75%"/>
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
 <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/src/resources/capstone2/images/test_image_examples.jpeg" />
 </p>

# CNN Model 1

I constructed a first convolutional neural network based off the keras blog post for "Building Powerful Image Classification Models Using Very Little Data." https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Code for this model can be found in resources as 'model1.py'. Note that this file uses a flow_from_directory image generator.

 <p align="center">
 <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/src/resources/capstone2/images/m1_summary.png" width="50%" height="50%"/>
 </p>
 
 <p align="center">
 <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/src/resources/capstone2/images/m1_loss_acc50epoch.jpeg" />
 </p>

This model has o.k. training and test accuracy and is doing slightly better than guessing for fire in images. See example images below with their corresponding label and if the model identified fire correctly. Evaluated on hold-out images results:

 * Loss: 0.63
 * Accuracy: 0.69

 <p align="center">
 <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/src/resources/capstone2/images/m1_predictions50epoch.jpeg" />
 </p>

This model's results become less reliable when we look at its confusion matrix:
```
[[ 59  93]
 [ 58 112]]
```

# CNN Model 2

The second convolutional neural network references a Tensorflow classification tutorial. https://www.tensorflow.org/tutorials/images/classification

Code for this model can be found in resources as 'model2.py'. This model was defined with a best-weights checkpoint, model save checkpoint, and early stop checkpoint. Note that this file uses a image_dataset_from_directory image generator. This same method is used in the Google Colab notebook.

 <p align="center">
 <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/src/resources/capstone2/images/m2_summary.png" width="50%" height="50%" />
 </p>
 
 <p align="center">
 <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/src/resources/capstone2/images/m2_loss_acc_es.jpeg" />
 </p>

This model performs much better than the first CNN model. Evaluated on hold-out images:

 * Loss: 0.19
 * Accuracy: 0.92
 
 <p align="center">
 <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/src/resources/capstone2/images/m2_predictions_es.jpeg" />
 </p>
 
However this model's confusion matrix also shows high false positives annd false negatives.

```
[[ 73  79]
 [ 65 100]]
```

**********************************************
# Next Steps
**********************************************
# Data Augmentation & Different Metric

As we saw above, measuring the convolutional neural networks by accuracy alone did not shed light on the high false positive and false negative rates. In fire detection in general, it would be more costly to have false negatives, i.e. saying there isn't a fire when there is in fact a fire raging. 

A better metric is recall, or the total true fire images divided by the true fire images added to the false negatives, or images with fire that the model incorrectly identified.

Also, the models may have seen nature (nonfire) augmented images, but it did not see any augmented fire images. Meaning the models are overtraining on those fire images. I would have liked to remedy this by augmenting both image classes, and oversampling/augmenting the lesser class more.

# Optimal Threshold Tuning and Transfer Learning

I have several options for making Model 2 more accurate:

 1. Tune the prediction threshold to yield the best true positive rate.
 2. Transfer learning from a more robustly trained neural network like ImageNet.

# Layer Visualization

Using the keras-vis package of tensorflow, I would like to visualize the last output layer to determine what Model 2 is picking up on in the images.

# Further Research

Fire detection is incredibly important and is becoming more pressing as we experience a cyclical routine to forest fires. For example, every year, 3 months of the year, California is on fire and thousands lose their homes. That smoke blows to other parts of the country and creates horrible living conditions so that people are locked inside or hospitalized because they cannot breath. I am interested in passing satellite images of Earth through a more robust convolutional neural networks to determine if there is in fact a fire. I would like to build upon that model so that it takes into account other landscape features including but not limited to: weather conditions, plant life, fire history, and surrounding infrastrucutre. This way we could determine areas of high risk and monitor them closely for signs of fire starting. 

Lastly, as I get more data science tools under my belt, I would like to use similar landscape features to predict fire spread.


