**********************************************
# Fire Risk Analysis
**********************************************

#### Author: Chelsea Zaloumis
#### Galvanize DSI Capstone 3
*Last update: 9/21/2020*

 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/cawildfire.jpeg" width="75%" height="75%"/>
 </p>
 
 The objective with this project is to analyze risk for fire in Northern and Southern California based off of environmental conditions and satellite imagery. My initial goal was to build three models for analyzing risk:
 
 1. A CNN based of Xception for smoke detection in satellite imagery.
 2. An XGBoost Classifier for fire based on conditions data.
 3. A RNN (LSTM) to determine current and future fire risk based off historical data.
 
 # Data
 
This project is comprised of two datasets, one containing satellite imagery of Southern and Northern California, and one containing California Iriigation Management Information System, CIMIS, conditions.
 
 I collected image data from the USDA Forest Service (https://fsapps.nwcg.gov/afm/imagery.php) with selenium using a Chrome Driver. Interested readers can view the source code at image_selenium.py. Image data consists of true color satellite imagery at a spatial resolution of 250 meters.
 Satellites/sensors and their correspdonging image band combination are listed below. More information on the Terra and Aqua satellites can be found here: https://oceancolor.gsfc.nasa.gov/data/aqua/
 
 * Aqua MODIS (Moderate Resolution Imaging Spectroradiometer) Corrected Reflectance, True Color Composite (Bands 1, 4, and 3)
 * Terra MODIS Corrected Reflectance, True Color Composite (Bands 1, 4, and 3)
 
 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/test_image_examples.jpeg" width="75%" height="75%"/>
 </p>

 
The conditions dataframe was downloaded from CIMIS California Department of Water Resources thanks to their weather stations: https://cimis.water.ca.gov/Default.aspx Readers can access the cleaned csv's in the data folder. The corresponding conditions_df.csv represents entries from 1/1/2018 to 9/13/2020 and has the following columns where "Target" represents a binary classification for fire or no fire. The Target column was obtained by merging Wikipedia tables listing California counties and cities with a CIMIS Station table, then merging the resulting dataframe with conditions_df(.csv).

Stn Id |	Stn Name |	CIMIS Region |	Date |	ETo (in) |	Precip (in) |	Sol Rad (Ly/day) |	Avg Vap Pres (mBars) |	Max Air Temp (F) |	Min Air Temp (F) |	Avg Air Temp (F) |	Max Rel Hum (%) |	Min Rel Hum (%) |	Avg Rel Hum (%) |	Dew Point (F) |	Avg Wind Speed (mph) |	Wind Run (miles) |	Avg Soil Temp (F) | Target
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
249	| Ripon	| San Joaquin Valley |	8/6/2020 |	0.25 |	0.0 |	680.0 |	18.3 |	96.3 |	51.7 |	72.8 |	99.0 |	46.0 |	66.0 |	60.9 |	4.2 |	100.3 |	70.3 |	0


Data preparation resulted in approximately 2,000 satellite images (heavily imbalanced with more foggy images than smoke images) and 127,138 rows of data.

# Xception

The image models used in this project base their improvements around improving recall. Recall was determined the most important metric because it encapsulates the models' abilities to determine fewer false negatives, ultimately a more costly endeavor (think not categorizing a satellite image as smoky when it is in fact smoky).

A baseline CNN was built with poor recall ( true positive smoke in images / (true positive smoke in images + false negatives ) ) with results outlined below:

 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/1cnn/model_loss_acc.jpeg" width="55%" height="55%"/> <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/1cnn/model_roccurve_1.jpeg" width="35%" height="35%"/>

The baseline model failed to categorize very smokey images.

 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/c3/images/1cnn/model_prediction_6.jpeg" width="55%" height="55%"/>
 </p>

I chose Xception based off its modified depthwise separable convolution which builds on chained inception modules with two key differences: 1. perform 1×1 convolution first then channel-wise spatial convolution and 2. there is no non-linearity intermediate activation. The first of which does not contribute much to Xception's improved model architecture, while the 2nd attributes improved accuracy. Xception's model architecture:

 <p align="center">
 <img src="https://miro.medium.com/max/1400/1*hOcAEj9QzqgBXcwUzmEvSg.png" width="75%" height="75%"/>
 </p>
