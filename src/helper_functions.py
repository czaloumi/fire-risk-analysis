import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import auc, roc_curve, roc_auc_score, classification_report, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix

def plot_subgroup_hist(df, class1, class2):
    '''
    Displays information of 2 classes of a data set
    
    PARAMETERS
    ----------
        df: dataframe
        class1: dataframe of 1 class
        class2: dataframe of 2nd class
    '''
    dim = len(df.columns)
    fig, axs = plt.subplots(3, int(dim/3), figsize=(12, 6))
    for i, col_name in enumerate(class1.columns):
        bins = np.linspace(df[col_name].min(), df[col_name].max(), 20)
        height, binz = np.histogram(class1[col_name], bins=bins, density=True)
        bp1 = axs[i%3][i//3].bar(bins[:-1], height, .5*(bins[1]-bins[0]),
                     alpha=0.5, label="Fire", color='r')
        height, binz = np.histogram(class2[col_name], bins=bins, density=True)
        bp2 = axs[i%3][i//3].bar(bins[:-1]+.5*(bins[1]-bins[0]), height,
                     .5*(bins[1]-bins[0]), color='b', alpha=.5)
        axs[i%3][i//3].set_title(col_name)
        axs[i%3][i//3].legend((bp1[0], bp2[0]), ("Fire", "No Fire"), loc='best')

    plt.tight_layout()

    return fig, axs

def plot_roc(ax, df):
    ax.plot([1]+list(df.fpr), [1]+list(df.tpr), label="ROC")
    ax.plot([0,1],[0,1], 'k', label="random")
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()

def roc_continuous_curve(probabilities, alpha, labels):
    probabilities = np.where(probabilities > alpha, 1, 0)
    
    df = pd.DataFrame({'probabilities': probabilities, 'y': labels})
    df.sort_values('probabilities', inplace=True)

    actual_p = df.y.sum()
    actual_n = df.shape[0] - df.y.sum()
    
    df['tn'] = (df.y == 0).cumsum()
    df['fn'] = df.y.cumsum()
    df['fp'] = actual_n - df.tn
    df['tp'] = actual_p - df.fn
    
    df['fpr'] = df.fp/(df.fp + df.tn)
    df['tpr'] = df.tp/(df.tp + df.fn)
    df['precision'] = df.tp/(df.tp + df.fp)
    df['F1'] = 2*((df.tp/(df.tp + df.fp)) * (df.tp/(df.tp + df.fn)))/((df.tp/(df.tp + df.fp)) + (df.tp/(df.tp + df.fn)))
    df = df.reset_index(drop=True)
    return df

def model_comparison(model_list, X_train, y_train, X_test, y_test):
    
    fig, ax = plt.subplots(figsize=(15, 10))

    for m in model_list:
        model = m
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        y_proba = y_proba[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Calculate Area under the curve to display on the plot
        area_under_curve = auc(fpr, tpr)

    # Plot the computed values
        plt.plot(fpr, tpr, label=f"{model.__class__.__name__}, {round(area_under_curve, 2)}")

    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve')
    plt.xlabel('1-Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.legend()
    plt.show();

def conf_matrix(estimator, X_train, y_train, X_test, y_test):
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    
    y_test_0 , y_test_1  = np.bincount(y_test)
    
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_pred)
    cfn_matrix = np.array([[tn, fp],[fn, tp]])
    
    cfn_norm_matrix = np.array([[1.0 / y_test_0, 1.0/ y_test_0],[1.0 / y_test_1, 1.0 / y_test_1]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig, axs = plt.subplots(2, figsize=(6,6))
    sns.heatmap(cfn_matrix, linewidths=0.5, annot=True, ax=axs[0])
    axs[0].set_title('Confusion Matrix')
    axs[0].set_ylabel('Real Classes')
    axs[0].set_xlabel('Predicted Classes')
    
    sns.heatmap(norm_cfn_matrix, linewidths=0.5, annot=True, ax=axs[1])
    axs[1].set_title('Normalized Confusion Matrix')
    axs[1].set_ylabel('Real Classes')
    axs[1].set_xlabel('Predicted Classes')
    
    plt.tight_layout()
    plt.show()
    
def cross_val(model, X_train, y_train, X_test, y_test, splits=5):
    '''
    Cross Validation function
    
    PARAMETERS
    ----------
        model: estimator
        X_train, y_train: training data, targets
        X_test, y_test: testing data, targets
        splits: kfolds
        
    RETURNS
    -------
        accuracy: float
        precision: float
        recall: float
        f1-score: float
    '''
    kf = KFold(n_splits=splits)
        
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for train_index, test_index in kf.split(X_train):
        X_train_split, X_test_split = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_split, y_test_split = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(X_train_split, y_train_split)
        pred = model.predict(X_test_split)

        assess = lambda method, val=y_test_split, pred=pred: method(val, pred)

        accuracy.append(assess(accuracy_score))
        precision.append(assess(precision_score))
        recall.append(assess(recall_score))
        f1.append(assess(f1_score))
        
    return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1)
    
def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
    Gridsearches to hypertune estimator on training data.
    
    PARAMETERS
    ----------
        estimator: the type of model (e.g. RandomForestRegressor())
        paramter_grid: dictionary defining the gridsearch parameters
        X_train: 2d numpy array
        y_train: 1d numpy array
        
    RETURNS
    -------
        best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=10,
                                    cv=3,
                                    scoring='recall')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_ 
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), 
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best

class Fire(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y)

    def predict(self, model):
        self.model = model
        kf = KFold(n_splits=5)
        
        accuracy = []
        precision = []
        recall = []
        f1 = []

        for train_index, test_index in kf.split(self.X_train):
          X_train_split, X_test_split = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
          y_train_split, y_test_split = self.y_train.iloc[train_index], self.y_train.iloc[test_index]
          self.model.fit(X_train_split, y_train_split)
          self.pred = self.model.predict(X_test_split)

          assess = lambda method, val=y_test_split, pred=self.pred: method(val, pred)

          accuracy.append(assess(accuracy_score))
          precision.append(assess(precision_score))
          recall.append(assess(recall_score))
          f1.append(assess(f1_score))
        
        return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1)
      
    def get_rates(self):
        self.proba = self.model.predict_proba(self.X_test)
        self.proba = self.proba[:,1]
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_test, self.proba)
        self.auc = auc(self.fpr, self.tpr)

        return self.fpr, self.tpr, self.auc

    def cm(self):
        return plot_confusion_matrix(self.model, self.X_test, self.y_test, cmap=plt.cm.Purples , normalize='true')
    
    def plot_roc(self, ax, model):
        if model == 'knn':
          ax.plot(self.fpr, self.tpr, color='orange', label=f'{model}: {round(self.auc, 4)}')
          return ax
        elif model == 'forest':
          ax.plot(self.fpr, self.tpr, color='green', label=f'{model}: {round(self.auc, 4)}')
          return ax
        elif model == 'boost1':
          ax.plot(self.fpr, self.tpr, color='red', label=f'{model}: {round(self.auc, 4)}')
          return ax
        else:
          ax.plot(self.fpr, self.tpr, color='purple', label=f'{model}: {round(self.auc, 4)}')
          return ax

    def plot_importance(self):
        return plot_importance(self.model, max_num_features=15)

def profit_curve(estimators, X_train, y_train, X_test, y_test):
    '''
    Based off user-defined prices for FP, FN, TP, TN
    function plots thresholds for positive class predictions
    and the corresponding cost/profit curve.
    
    PARAMETERS
    ----------
        estimators: list of models
        X_train: (n, m) training dataframe
        y_train: (n,) training labels
        X_test: (q, m) testing dataframe
        y_test: (q,) testing labels
    
    RETURNS
    -------
        Profit curve
    '''
    
    # User input() for costs
    FP_price = float(input('Price of False Positive:'))
    FN_price = float(input('Price of False Negative:'))
    TP =  float(input('Price of True Positive:'))
    TN =  float(input('Price of True Negative:'))
    
    # Utility matrix
    utility_M = np.array([[TP, FP_price], [FN_price, TN]])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for m in estimators:
        m.fit(X_train, y_train)
        
        # Predictions probablies
        proba = m.predict_proba(X_test)
        proba = proba[:,1]
        
        # List of thresholds
        thresholds = np.linspace(1,0,50)
        
        # Costs for each threshold
        costs = []
 
        for thresh in thresholds:
            tp, fp, tn, fn = 0, 0, 0, 0
            
            # Calculate prediction at each threshold and find number of TP, FP, FN, TN
            for i, (y_pred, prob) in enumerate(zip(y_test, proba)):
                if y_pred == 1:
                    if prob > thresh:
                        tp += 1
                    else:
                        fn += 1
                elif y_pred ==0:
                    if prob > thresh:
                        fp += 1
                    else:
                        tn +=1
            
            # Make cm and find cost
            # Add cost to empty cost list
            conf_matrix = np.array([[tp, fp], [fn, tn]])
            norm_cm = conf_matrix / (np.sum(conf_matrix))
            cost = np.sum(norm_cm * utility_M)
            costs.append(cost)
        
        # Sort thresholds
        best_thresh = thresholds[np.argmax(costs)]
        
        ax.plot(thresholds, costs, label=f'{m.__class__.__name__}:  ${max(costs):.1f} at {best_thresh:.4f}')
        ax.axvline(best_thresh, min(costs), max(costs), linestyle='--', color=(.6,.6,.6))
        ax.axhline(max(costs), -.03, best_thresh, linestyle='--', color=(.6,.6,.6))
        ax.set_ylabel('Cost / Profit', size=14)
        ax.set_xlabel('Threshold for Positive Classification', size=14)
        ax.set_title('Profit Curve', size=16)
        #plt.text(best_thresh+.01, max(cost_list)/2 ,s=f'{best_thresh:.4f}')
        ax.set_xlim(-.03, 1.005)
        ax.set_ylim(min(costs), 2+max(costs))
        ax.legend(loc='best')
        #plt.savefig('../images/profit_curves.png')

    return best_thresh

def plot_feature_importance(importances, features, model_title, N=15):
    '''
    Plots top N feature importances for tree based models.
    
    PARAMETERS
    ----------
        importances: model.feature_importances_
        features: df.columns
        model_title: string for plot title
        N: int default 15 top features
        
    RETURNS
    -------
        Plot of feature importances
    '''
    feature_importance = np.array(importances)
    feature_names = np.array(features)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'][:N], y=fi_df['feature_names'][:N])
    plt.title(model_title + ' Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')