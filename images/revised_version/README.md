Updated 8/30/2020

Images for new README to replace initial project submitted 8/28/2020.

**********************************************
# Forest Fire Detection
**********************************************

#### Author: Chelsea Zaloumis
#### Galvanize DSI Capstone 2
*Last update: 8/30/2020*

![title](images/cawildfire.jpeg)

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


