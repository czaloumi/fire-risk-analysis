import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_curve, auc, classification_report, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix

from helper_functions import *

# Load data and clean
df = pd.read_csv('../data/conditions_df.csv')
to_drop = ['Date', 'Stn Id', 'Stn Name', 'CIMIS Region', 'Notes']
df.drop(to_drop, axis=1, inplace=True)
df = df.fillna(df.mean())
scaled_df = StandardScaler().fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

y = df.pop('Target')

X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, stratify=y, test_size=.2)

# Tune RF
rf_parameter_grid = {
    'max_depth': [10, 30, 50, 70],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [2, 4],
    'min_samples_split': [2, 5],
    'n_estimators': [200, 400, 500]
}

rf_params, rf_gridsearch = gridsearch_with_output(RandomForestClassifier(class_weight='balanced'), rf_parameter_grid, X_train, y_train)

filename = 'tuned-rf.pkl'
pickle.dump(rf_gridsearch, open(filename, 'wb'))

# Tune XGBoost
xgb_parameter_grid = {
                    'max_depth': [3, 9],
                    'n_estimators': [100, 800, 900], 
                    'learning_rate': [0.01, 0.1],
                    'lambda': [0.5, 0.8], # l2 regualrization
                    'alpha': [0.25, 0.5]  # l1 regularization
                    }

xgb_params, xgb_gridsearch = gridsearch_with_output(XGBClassifier(class_weight='balanced'), xgb_parameter_grid, X_train, y_train)

filename = 'tuned-xgb.pkl'
pickle.dump(xgb_gridsearch, open(filename, 'wb'))