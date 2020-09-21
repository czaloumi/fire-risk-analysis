from flask import render_template, request, jsonify
from flask import Flask
import pandas as pd
import numpy as np
import pickle
import xgboost
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from tkcalendar import DateEntry

app = Flask(__name__)


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
    boost = pickle.load(open("../../src/pima.pickle.dat", "rb"))
    xception = load_model('../../src/xception_50epoch.h5')
    
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
    
    # Predict fire risk based off conditions
    conditions_y = df.pop('Target')
    conditions_y_pred = boost.predict(df)
    conditions_risk = np.mean(conditions_y_pred)
    
    # Overall risk as weighted of Xception prediciton & XGBoost prediction
    risk = .8*(image_risk.ravel()[0]) + 0.2*conditions_risk

    return render_template('predictions.html', region=region, date=date, risk=risk)

@app.route('/', methods =['GET','POST'])    
def index():
    dropdown_list = ['norcal', 'socal']
    return render_template('home.html', dropdown_list=dropdown_list)


@app.route('/risk', methods=['GET','POST'])
def predict():
    button = request.form.get('value')
    if button == 'Fire away':
        region = request.select.get('value')
        region = str(region)
        date = request.form['date']
        combinedModels(region, date)



if __name__ == "__main__":
    app.debug=True
    app.run()