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

The past two weeks left many states in smoke and I was encouraged to begin data science exploration into fire detection and eventually fire prediction. My goal in this project is to create a convolutional neural network to accurately identify forest fires in images.

# Data

Kaggle fire images dataset: https://www.kaggle.com/phylake1337/fire-dataset
 * 2 folders
 * fireimages folder contains 755 outdoor-fire images some of them contains heavy smoke
 * non-fireimages folder contains 244 nature images (eg: forest, tree, grass, river, people, foggy forest, lake, animal, road, and waterfall)

# Image Preprocessing

The initial dataset is imbalanced. To remedy this, we have several options:

  * Web scrap/download additional nature images from the internet
  * Generate augmented images of the nature images already in the dataset
  * Undersample from the fire class of images 
  
We will generate augmented images. In the resources folder, readers can find 'gen_images.py' by which they can generate augmented images for either fire or non fire classes. This file assumes there is a path to one data folder containing both fire_images and non_fire_images subfolders and that the image formats are .png.

Examples of images comprising the dataset:
 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/test_image_examples.jpeg" />
 </p>

# CNN Model 1

I constructed a first convolutional neural network based of this keras blog post: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Code for this model can be found in resources as 'model1.py'. Note that this file uses a flow_from_directory image generator like the blog post.

 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m1_summary.png" width="50%" height="50%"/>
 </p>
 
 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m1_loss_acc50epoch.jpeg" />
 </p>

This model has o.k. training and test accuracy and is doing slightly better than guessing for fire in images. See example images below with their corresponding label and if the model identified fire correctly.

 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m1_predictions50epoch.jpeg" />
 </p>

# CNN Model 2

The second convolutional neural network references a Tensorflow classification tutorial: https://www.tensorflow.org/tutorials/images/classification

Code for this model can be found in resources as 'model2.py'. Note that this file uses a image_dataset_from_directory image generator like the blog post. This method is followed in the Google Colab notebook for following along.

 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/old/m2_summary.png" />
 </p>
 
 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m2_loss_acc.jpeg" />
 </p>

This mode performs much better than the first CNN model and accurately identifies the two image classes as illustrated below.

 <p align="center">
 <img src="https://github.com/czaloumi/fire/blob/master/images/revised_version/m2_predictions.jpeg" />
 </p>

# ROC Curves & Confusion Matrix

