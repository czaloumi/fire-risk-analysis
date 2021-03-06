**********************************************
# Fire Risk Analysis
**********************************************

#### Author: Chelsea Zaloumis
*Last update: 1/24/2020*

 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/cawildfire.jpeg" width="75%" height="75%"/>
 </p>
2020 fires in California are breaking records and are the worst the state has seen in the past 18 years. Tens of thousands have been uprooted from their homes and billows of unhealthy air conditions have blanketed the rest of California's neighbor states.

The objective of this project is to analyze risk for fire in Northern and Southern California from 1/1/2018 to 9/13/2020 based off of environmental conditions and satellite imagery. My objective can be organized in three points:
 1. Transfer learning with Xception for smoke detection in satellite imagery.
 2. An XGBoost Classifier for fire risk based on daily conditions data.
 3. Ensemble models, weight probabilities, and deploy on Flask APP.
 
 # Data
This project is comprised of two datasets, one containing daily satellite imagery of Northern and Southern California, and one of daily environmental conditions by county and region. Data preparation resulted in approximately 2,000 satellite images (heavily imbalanced with more foggy images than smoke images) and 127,138 observations with 14 features of environmental conditions (also heavily imbalanced). I scraped data from the two websites below.
 * USDA Forest Service Satellite Imagery: https://fsapps.nwcg.gov/afm/imagery.php
 * CIMIS Conditions: https://cimis.water.ca.gov/Default.aspx
 
## Satellite Images
I collected image data from the USDA Forest Service (https://fsapps.nwcg.gov/afm/imagery.php) with selenium using a chromedriver. Interested readers can view the source code in image_selenium.py. Image data consists of true color satellite imagery at a spatial resolution of 250 meters. Satellites/sensors and their correspdonging image band combination are listed below. More information on the Terra and Aqua satellites can be found here: https://oceancolor.gsfc.nasa.gov/data/aqua/

 * Aqua MODIS (Moderate Resolution Imaging Spectroradiometer) Corrected Reflectance, True Color Composite (Bands 1, 4, and 3)
 * Terra MODIS Corrected Reflectance, True Color Composite (Bands 1, 4, and 3)

Images were then filed into smoke and fog subfolders and labeled with date and region (norcal or socal). Here are two examples of the image classes. The left image is labeled as smoke and visually the smoke appears off-color and does not have a general pattern or specific density. The image on the right is labeled as fog and we can see the differences in fog vs. smoke off the bat: fog is whiter, has patterns, or appears in dense clouds.
 
 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/2020-09-05_1.jpg" width="50%" height="50%"/><img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/2018-01-05_1.jpg" width="50%" height="50%"/>
 </p>

## Environmental Conditions Data
The conditions dataframe was downloaded in bathces from CIMIS California Department of Water Resources which provides hourly, daily, and monthly information. I chose daily data entries (https://cimis.water.ca.gov/Default.aspx). Readers can access the cleaned csv (conditions_df.csv) in the data folder. The data represents entries from 1/1/2018 to 9/13/2020 and has the following columns where "Target" represents a binary classification for fire or no fire. The Target column was obtained by merging Wikipedia tables listing California fires by county and city with a CIMIS Station table, then merging the resulting dataframe with conditions_df(.csv).

There were approximately 16% null values of the positive target class observations. KNN Imputation was used to determine what to fill nans with.

<img src='images/eda_histograms.png'>

# Xception
I will leave it to readers to familiarize themselves with Xception. The image model to detect smoke was built using transfer learning with Xception. Layers were unfrozen in 4 layer increments and trained for 10 epochs until the final model had approximately 74 unfrozen layers. Note that the imbalanced dataset needed to be weighted before training. Xception's model architecture:
 <p align="center">
 <img src="https://miro.medium.com/max/1400/1*hOcAEj9QzqgBXcwUzmEvSg.png" width="75%" height="75%"/>
 </p>
The final model obtained **95% accuracy, 76% recall, 89% precision**. Several predictions are listed below, to evaluate the model's abilities to determine smoke in images it was not trained on. The first image is labeled as smoke and the model classified it correctly as smoke. The second is labeled fog and the model also classified it correctly. The third image is a great example for how the model is not perfect and categorized the image as mainly smoke. Visually, it is hard for the human eye to determine if there is smoke or fog in that image. 
 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/3xception_70trained/xception_metrics1.png" width="175%" height="175%"/>
 </p>
 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/3xception_70trained/confusion_matrixsmoke%20classification.png" width="50%" height="50%"/>
 </p>
  <p align="center">
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/3xception_70trained/one_xception_prediction0.png" width="100%" height="100%"/>
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/3xception_70trained/one_xception_prediction4.png" width="100%" height="100%"/> 
   <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/3xception_70trained/one_xception_prediction3.png" width="100%" height="100%"/> 
   </p>

# XGBoost Classifier
## Model Comparison
Compared out of box Logistic Regression, Decision Tree, Random Forest, and XGBoost. Random Forest performed best. Hypertuned Random Forest and XGBoost using recall for scoring metric. Recall is the ideal metric for fire risk because a false negative would result in a high fire risk day going unnoticed. Random Forest still outperformed XGBoost after gridsearching optimal parameters.
  <p align="center">
  <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/images/0conditions_df/model-comparison.png" width="70%" height="70%"/>
  </p>

Tuned XGBoost compared to out of box. It's important to note that the hypertuned boost performed *worse* than out of the box... Tuned RF compared to out of box.
  <p align="center">
  <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/images/0conditions_df/xgb-comparison.png" width="70%" height="70%"/>
  <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/images/0conditions_df/rf-comparison.png" width="70%" height="70%"/>
  </p>

---
*Not updated past this point*
---

## Final XGBoost Feature Importances
The hyptertuned XGBoost's results and feature importances as determined by gain. It is important to note the use of gain in the feature importances compared to other feature importance metrics. ‘Gain’ is the improvement in accuracy brought by a feature to the branches it is on. These are interesting results to note what environmental conditions contribute to a more accurate predicion by this model.
  <p align="center">
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/boost_cm.jpeg" width="45%" height="45%"/><img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/0conditions_df/boost2_gain.png" width="50%" height="50%"/>
  </p>
Exploring the top two important features, here are their partial dependence plots portraying their affects on fire risk. Based off the first partial dependence plot for average temperatue we can see that the hotter, the greater risk for fire, until ~75 degrees. The reader should notice that the risk only increases about 15%. Solar radiation is linearly related to fire risk up until a peak of ~450 langley/day at which the risk begins decreasing. The third partial dependence plot shows us that for average soil temperature less than ~62 degrees, fire risk is independent of solar radiation. But for average soil temperature greater than 62 degrees, there is a strong dependence on solar radiation.
  <p align="center">
  <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/images/0conditions_df/partial-dependence.png" width="75%" height="75%"/>
  </p>
  
# Combining Models
Once my models were up to par, I spent a long time attempting to build a new model which combined the two using keras's functional api. I was unable to modify inputs and outputs to get a functional api model working and instead combined model predictions for fire risk by weighting the two model predictions.

Users can view the combined model's fire risk analysis soon to be deployed on an AWS EC2 instance. The combined models predict fire risk in the past (1/1/2018-9/13/2020) by entering a region: 'norcal' or 'socal' and a corresponding date. The model then outputs the risk for fire on that day given the amount of smoke detected in the satellite image and the risk for fire predicted given the day's conditions.
  <p align="center">
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/flask_app1.png" width="100%" height="100%"/>
  <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/flask_app2.png" width="100%" height="100%"/>
  </p>

# Next Steps
## LSTM
This project has only scraped the surface for fire prevention and risk analysis. In order to analyze current and future risk, it is necessary to build a model with a memory component. One such model is a long short-term memory model, LSTM.

At this time, I have built a basic LSTM for one weather station, for one month, with one feature (Avg Soil Temp (F)-deemed most important feature for gain by XGBoost Classifier) however the problem is much more complex with the conditions dataframe on hand. The conditions dataframe has 14 features, and 126 weather stations with up to 900 or more repeated dates. This problem will grow more complex with more data, and adding more data is necessary to maintain relevance.

Ideas for moving forward:
* build a model trained for all features, for one weather station (region/location) for a year or less time.
* build a model trained for the most important features, for one weather station, for a year or less time.

# References
  <p align="center">
  <img src="https://github.com/czaloumi/fire-risk-analysis/blob/master/images/tech-used.png" width="100%" height="100%"/>
  </p>
