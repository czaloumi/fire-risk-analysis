**********************************************
# Fire Risk Analysis
**********************************************

#### Author: Chelsea Zaloumis
#### Galvanize DSI Capstone 3
*Last update: 9/23/2020*

 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/cawildfire.jpeg" width="75%" height="75%"/>
 </p>
 
 The objective of this project is to analyze risk for fire in Northern and Southern California based off of environmental conditions and satellite imagery. My initial goal was to build three models for analyzing risk:
 
 1. Transfer learning Xception for smoke detection in satellite imagery.
 2. An XGBoost Classifier for fire risk based on daily conditions data.
 3. A RNN and LSTM to determine current and future fire risk based off historical data.
 
 # Data
 
This project is comprised of two datasets, one containing daily satellite imagery of Northern and Southern California, and one of daily environmental conditions by county and region. Data preparation resulted in approximately 2,000 satellite images (heavily imbalanced with more foggy images than smoke images) and 127,138 rows of data.
 * USDA Forest Service Satellite Imagery: https://fsapps.nwcg.gov/afm/imagery.php
 * CIMIS Conditions: https://cimis.water.ca.gov/Default.aspx
 
## Satellite Images
I collected image data from the USDA Forest Service (https://fsapps.nwcg.gov/afm/imagery.php) with selenium using a chromedriver. Interested readers can view the source code at image_selenium.py. Image data consists of true color satellite imagery at a spatial resolution of 250 meters. Satellites/sensors and their correspdonging image band combination are listed below. More information on the Terra and Aqua satellites can be found here: https://oceancolor.gsfc.nasa.gov/data/aqua/

 * Aqua MODIS (Moderate Resolution Imaging Spectroradiometer) Corrected Reflectance, True Color Composite (Bands 1, 4, and 3)
 * Terra MODIS Corrected Reflectance, True Color Composite (Bands 1, 4, and 3)

Images were then filed into smoke and fog subfolders and labeled with date and region (norcal or socal). Here are two examples of the image classes. The left image is labeled as smoke and visually the smoke appears off-color and does not have a general pattern or specific density. The image on the right is labeled as fog and we can see the differences in fog vs. smoke off the bat: fog is whiter, has patterns, or appears in dense clouds.
 
 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/2020-09-05_1.jpg" width="50%" height="50%"/><img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/2018-01-05_1.jpg" width="50%" height="50%"/>
 </p>

## Environmental Conditions Data
The conditions dataframe was downloaded from CIMIS California Department of Water Resources which provides hourly, daily, and monthly information. I chose daily data entries (https://cimis.water.ca.gov/Default.aspx). Readers can access the cleaned csv's in the data folder. The corresponding conditions_df.csv represents entries from 1/1/2018 to 9/13/2020 and has the following columns where "Target" represents a binary classification for fire or no fire. The Target column was obtained by merging Wikipedia tables listing California counties and cities with a CIMIS Station table, then merging the resulting dataframe with conditions_df(.csv).

Stn Id |	Stn Name |	CIMIS Region |	Date |	ETo (in) |	Precip (in) |	Sol Rad (Ly/day) |	Avg Vap Pres (mBars) |	Max Air Temp (F) |	Min Air Temp (F) |	Avg Air Temp (F) |	Max Rel Hum (%) |	Min Rel Hum (%) |	Avg Rel Hum (%) |	Dew Point (F) |	Avg Wind Speed (mph) |	Wind Run (miles) |	Avg Soil Temp (F) | Target
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
249	| Ripon	| San Joaquin Valley |	8/6/2020 |	0.25 |	0.0 |	680.0 |	18.3 |	96.3 |	51.7 |	72.8 |	99.0 |	46.0 |	66.0 |	60.9 |	4.2 |	100.3 |	70.3 |	0

# Xception

The image models used in this project measure their improvements around recall. Recall was determined the most important metric because it encapsulates the models' abilities to determine fewer false negatives, ultimately a more costly endeavor (think not categorizing a satellite image as smoky when it is in fact smoky).

A baseline CNN was built with poor recall ( true positive smoke in images / (true positive smoke in images + false negatives ) ) with results outlined below. The baseline model failed to categorize very smokey images.

 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/1cnn/model_loss_acc.jpeg" width="55%" height="55%"/> <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/1cnn/model_roccurve_1.jpeg" width="35%" height="35%"/>
 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/1cnn/model_prediction_6.jpeg" width="55%" height="55%"/>
 </p>
To improve on the baseline model, I chose to transfer learn with Xception based off its modified depthwise separable convolution which builds on chained inception modules with two key differences: 1. perform 1×1 convolution first then channel-wise spatial convolution and 2. there is no non-linearity intermediate activation. The first of which does not contribute much to Xception's improved model architecture, while the 2nd attributes improved accuracy. The final Xception model has approximately 74 unfrozen layers that were slowly trained in 4 layer increments for 10 epochs. Xception's model architecture:
 <p align="center">
 <img src="https://miro.medium.com/max/1400/1*hOcAEj9QzqgBXcwUzmEvSg.png" width="75%" height="75%"/>
 </p>
Intermediate metric measurements while training and unfreezing layers show Xception already surpassing the baseline model's metrics. The final trained model's confusion matrix is displayed below. Final prediction examples on satellite imagery displayed below as well.
 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/3xception_70trained/xception_metrics1.png" width="175%" height="175%"/>
 </p>
 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/3xception_70trained/confusion_matrixsmoke%20classification.png" width="50%" height="50%"/>
 </p>
  <p align="center">
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/3xception_70trained/one_xception_prediction0.png" width="100%" height="100%"/>
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/3xception_70trained/one_xception_prediction1.png" width="100%" height="100%"/> 
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/3xception_70trained/one_xception_prediction4.png" width="100%" height="100%"/> 
   </p>

# XGBoost Classifier
## Model Comparison
The conditions_df was modified further to fit classifier models on. Modifications included dropping time, station name, and CIMIS region. Prior to deciding on xgboost, knn and random forest classifiers were compared. Because random forest and xgboost performed best, a baseline xgboost was built without hypertuning and obtained the following results. A gridsearch was conducted on an AWS EC2 instance to hypertune xgboost. The grid search was ran twice for approximately 9 hours each run.
  <p align="center">
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/0conditions_df/knn_cm.jpeg" width="30%" height="30%"/><img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/0conditions_df/forest_cm.jpeg" width="30%" height="30%"/><img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/0conditions_df/boost_cm.jpeg" width="30%" height="30%"/>
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/0conditions_df/run1_roccurves%2Cjpeg.png" width="35%" height="35%"/><img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/0conditions_df/boost_comparison_roc.jpeg" width="40%" height="40%"/>
  </p>
## Final XGBoost
The hyptertuned xgboost's results and feature importances as determined by gain. It is important to note the use of gain in the feature importances compared to other feature importance metrics. ‘Gain’ is the improvement in accuracy brought by a feature to the branches it is on. These are interesting results to note what environmental conditions contribute to a more accurate fire prediction.
  <p align="center">
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/0conditions_df/boost2_cm.jpeg" width="45%" height="45%"/><img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/0conditions_df/boost2_gain.png" width="50%" height="50%"/>
  </p>

# Combining Models
Once my models were up to par, I spent a long time attempting to build a new model which combined the two using keras's functional api. I was unable to modify inputs and outputs to get a functional api model working and instead combined model predictions for fire risk by weighting the two model predictions.

Users can view the combined model's fire risk analysis soon to be deployed on an AWS EC2 instance. The combined models predict fire risk in the past (1/1/2018-9/13/2020) by entering a region: 'norcal' or 'socal' and a corresponding date. The model then outputs the risk for fire on that day given the amount of smoke detected in the satellite image and the risk for fire predicted given the day's conditions.
  <p align="center">
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/flask_app1.png" width="100%" height="100%"/>
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/flask_app2.png" width="100%" height="100%"/>
  </p>

# RNN & Next Steps
Unfortunately at this time I have not been able to get a RNN and LSTM working for a dataframe with multiple entries on the same date for more than one weather station. Moving forward, after training a RNN/LSTM to predict accurate fire risk given the previous few day's conditions, I will be able to predict today or the next few day's risk for fire.
