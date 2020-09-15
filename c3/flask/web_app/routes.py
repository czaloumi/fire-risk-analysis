from flask import render_template, request, jsonify
from web_app import app
import pickle
import pandas as pd
import numpy as np

#import pickle
#with open('path/to/df.pkl', 'rb') as f:
    #df=pickle.load(f)

#features = ['list_of_strings_of_features']
#X=df[features].values

#links = {'dict_of_links'}

# Decorators - modify functions that follows it
# when a web browser requests either these two URLs, 
# Flask will invoke this function and return value back to the browers
@app.route('/')     # associates the URLS to this function

@app.route('/index')
def index():
    user = {'username': 'Test User'}
    return render_template('index.html', title='Home', user=user)

if __name__ == "__main__":
    app.debug=True
    app.run()