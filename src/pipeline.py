import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_and_clean(path, drop_dates=True, drop_min_max=True, drop_cat=False):
    '''
    Loads conditions dataframe, drops min/max columns,
    several categorical columns, and one outlier.
    
    PARAMETERS
    ----------
        path: string to conditions_df
        drop_min_max: boolean for dropping min max columns where avgs already exist
        drop_dates: boolean for dropping dates if not timeseries problem
        drop_cat: boolean for dropping 'Stn Id'
        
    RETURNS
    -------
        Cleaned dataframe
        Dictionary station name -> region mapping
    '''
    df = pd.read_csv(path)
    
    # Removing one outlier
    df = df[df['Avg Wind Speed (mph)'] > 0]

    if drop_dates:
        df.drop('Date', axis=1, inplace=True)
        
    if drop_min_max:
        df.drop(['Max Air Temp (F)', 'Max Rel Hum (%)', 'Min Air Temp (F)',
           'Min Rel Hum (%)'], axis=1, inplace=True)
    
    if drop_cat:
        df.drop(['Stn Id', 'Stn Name', 'CIMIS Region', 'Notes'], axis=1, inplace=True)
        
        return df
    else:
        # name -> region dict
        name_region_map = {}
        for stn_id, stn_name in zip(np.unique(df['Stn Id']), np.unique(df['Stn Name'])):
            name_region_map[stn_id] = [stn_name, np.unique(df[df['Stn Name']==stn_name]['CIMIS Region'])[0]]
    
        to_drop = ['Stn Name', 'CIMIS Region', 'Notes']
        df.drop(to_drop, axis=1, inplace=True)

        return df, name_region_map

def pipeline(data_path, drop_cat=True):
    '''
    Pipeline for modeling.
    Loads data with load_and_clean,
    imputes missing values with KNNImputer,
    standardizes numerical features,
    if 'Stn Id' column is kept, one-hot-encodes.

    PARAMETERS
    ----------
        data_path: string path to data csv
        drop_cat: boolean for if 'Stn Id' is dropped (default)

    RETURNS
    -------
        dataframe ready for modeling
    '''
    if drop_cat:
        df = load_and_clean(data_path, drop_cat=True)
        target = df.pop('Target')
        df_num = df.copy()
    else:
        df, region_dict = load_and_clean(data_path)
        target = df.pop('Target')
        df_num = df.drop('Stn Id', axis=1)

    num_pipeline = Pipeline([
        ('imputer', KNNImputer(weights='distance')),
        ('scaler', StandardScaler())
    ])

    num_attribs = list(df_num.columns)
    
    if drop_cat:
        fire_prepared = num_pipeline.fit_transform(df)
        cols = num_attribs    

        df = pd.DataFrame(fire_prepared, columns=cols)    
    else:
        cat_attribs = ['Stn Id']

        full_pipeline = ColumnTransformer([
            ('num', num_pipeline, num_attribs),
            ('cat', OneHotEncoder(), cat_attribs)
        ])
        
        fire_prepared = full_pipeline.fit_transform(df)

        cat_cols = list(full_pipeline.named_transformers_['cat'].categories_[0])
        station_cols = []
        for col in cat_cols:
            station_cols.append(region_dict[col][0])
            
        cols = list(df_num.columns) + station_cols

        df = pd.DataFrame(fire_prepared.todense(), columns=cols)
    
    df['Target'] = target
    df.Target = df.Target.fillna(0)

    if full_pipeline:
        return df, full_pipeline

    else:
        return df, num_pipeline

if __name__=="__main__":
    path = '../data/stratified_train.csv'
    df, pipeline = pipeline(path)
    
    print(df.shape)
    print(df.Target.value_counts() / len(df))
    print(df.head())