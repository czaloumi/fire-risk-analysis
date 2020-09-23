from flask import Flask
from flask import render_template, request, jsonify
from flask import Response
import pandas as pd
import numpy as np
import glob
import pickle
import xgboost
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
plt.switch_backend('Agg')
plt.style.use('seaborn-whitegrid')
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model




app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def combinedModels(region, date):
    '''
    Combines trained Xception XGBClassifier
    returns weighted prediction.
    PARAMETERS
    ----------
    region: 'norcal' or 'socal'; string
    date: b/t 1/1/2018 and 9/13/2020; string
    RETURNS
    -------
    weighted prediction for fire risk
    '''

    # Load models
    boost = pickle.load(open("../../models/xgboost_model", "rb"))
    xception = load_model('../../models/xception_50epoch.h5')
    
    # Define Xception inputs
    batch_size = 16
    img_height = 256
    img_width = 256
    epochs=50
    
    if region == 'norcal':
        # Load norcal dataframe & clean
        norcal = pd.read_csv('../../data/norcal.csv')
        norcal = norcal.loc[:, ~norcal.columns.str.contains('^Unnamed')]
        df = norcal[norcal['Date']==date]
        df.drop('Date', axis=1, inplace=True)

        # reformat date: 9/1/2020 -> 2020-09-01
        datetime = pd.to_datetime(date, format='%m/%d/%Y')
        datetime = datetime.strftime('%Y-%m-%d')

        # find image by date in cleaned_data files
        path = glob.glob(f'../../data/cleaned_data/*/{datetime}_1.jpg')
        path = ''.join(element for element in path)
        image = tf.keras.preprocessing.image.load_img(path, interpolation="bilinear")
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.

        # Xception prediction for smoke in iamge
        image_risk = xception.predict(input_arr,batch_size=batch_size)
        
        # Gather image labels
        label =["FOG" if "fog_images" in path else "SMOKE"]
        predicted_label =["SMOKE" if image_risk > 0.5 else "FOG"] 

    elif region == 'socal':
        # Load norcal dataframe & clean
        socal = pd.read_csv('../../data/socal.csv')
        socal = socal.loc[:, ~socal.columns.str.contains('^Unnamed')]
        df = socal[socal['Date']==date]
        df.drop('Date', axis=1, inplace=True)
        
        # reformat date: 9/1/2020 -> 2020-09-01
        datetime = pd.to_datetime(date, format='%m/%d/%Y')
        datetime = datetime.strftime('%Y-%m-%d')

        # find image by date in cleaned_data files
        path = glob.glob(f'../../data/cleaned_data/*/{datetime}.jpg')
        path = ''.join(element for element in path)
        image = tf.keras.preprocessing.image.load_img(path, interpolation="bilinear")
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        
        # Xception prediction for smoke in iamge
        image_risk = xception.predict(input_arr,batch_size=batch_size)

        # Gather image labels
        label =["FOG" if "fog_images" in path else "SMOKE"]
        predicted_label =["SMOKE" if image_risk > 0.5 else "FOG"]
    
    # Predict fire risk based off conditions
    conditions_y = df.pop('Target')
    conditions_y_pred = boost.predict(df)
    conditions_risk = round(np.mean(conditions_y_pred),2)
    
    # Overall risk as weighted of Xception prediciton & XGBoost prediction
    if image_risk > 0.5:
        risk = round((0.8*(image_risk.ravel()[0]) + 0.2*conditions_risk), 2)
    elif conditions_risk > 0.5:
        risk = round((0.2*(image_risk.ravel()[0]) + 0.8*conditions_risk), 2)
    else:
        risk = round((0.5*(image_risk.ravel()[0]) + 0.5*conditions_risk), 2)

    return risk, image_risk, path, label, predicted_label, conditions_risk, df

@app.route('/', methods =['GET','POST'])    
def index():
    dropdown_list = ['norcal', 'socal']
    return render_template('home.html', dropdown_list=dropdown_list)


@app.route('/predict', methods=['GET','POST'])
def predict():
    region = request.form['region']
    region = str(region)
    date = request.form['date']
    date = pd.to_datetime(date, format='%Y-%m-%d')
    date = date.strftime('%-m/%-d/%Y')

    risk, prediction, path, label, predicted_label, conditions_risk, df = combinedModels(region, date)

    # plots image prediction
    img=mpimg.imread(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    ax1.imshow(mpimg.imread(path))
    ax1.set_title("label  {} : prediction {:0.2f}".format(label[0], prediction[0][0] ))
    ax1.grid(False)
    ax2.bar([0,1],[1-prediction[0][0],prediction[0][0]], color='orange')
    plt.xticks([0,1], ('FOG', 'SMOKE'))
    fig.savefig(f"./static/one_xception_prediction{prediction[0][0]}.png") 

    return render_template('predict.html', region=region, date=date, risk=risk, url =f"./static/one_xception_prediction{prediction[0][0]}.png", conditions_risk=conditions_risk, data=df.to_html())

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == "__main__":
    app.debug=True
    app.run()