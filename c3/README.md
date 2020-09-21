**********************************************
# Fire Risk Analysis
**********************************************

#### Author: Chelsea Zaloumis
#### Galvanize DSI Capstone 3
*Last update: 9/21/2020*

 <p align="center">
 <img src="https://github.com/czaloumi/cnn-fire-detection/blob/master/images/cawildfire.jpeg" width="75%" height="75%"/>
 </p>
 
 The objective with this project is to analyze risk for fire in Northern and Southern California based off of environmental conditions and satellite imagery.
 
 # Data
 
 I collected image data from the USDA Forest Service (https://fsapps.nwcg.gov/afm/imagery.php) with selenium using a Chrome Driver. Interested readers can view the source code at image_selenium.py. Image data consists of true color satellite imagery at a spatial resolution of 250 meters.
 Satellites/sensors and their correspdonging image band combination are listed below. More information on the Terra and Aqua satellites can be found here: https://oceancolor.gsfc.nasa.gov/data/aqua/
 * Aqua MODIS (Moderate Resolution Imaging Spectroradiometer) Corrected Reflectance, True Color Composite (Bands 1, 4, and 3)
 * Terra MODIS Corrected Reflectance, True Color Composite (Bands 1, 4, and 3)
 
