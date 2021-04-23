import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_and_clean(path, drop_min_max=True, keep_dates=True):
    '''
    Loads conditions dataframe, drops min/max columns,
    several categorical columns, and one outlier.
    
    PARAMETERS
    ----------
        path: string to conditions_df
        drop_min_max: boolean
        keep_dates: boolean
        
    RETURNS
    -------
        Cleaned dataframe
        Dictionary station name -> region mapping
    '''
    df = pd.read_csv(path)
    
    if drop_min_max:
        df.drop(['Max Air Temp (F)', 'Max Rel Hum (%)', 'Min Air Temp (F)',
           'Min Rel Hum (%)'], axis=1, inplace=True)
    
    # name -> region dict
    name_region_map = {}
    for stn_id, stn_name in zip(np.unique(df['Stn Id']), np.unique(df['Stn Name'])):
        name_region_map[stn_id] = [stn_name, np.unique(df[df['Stn Name']==stn_name]['CIMIS Region'])[0]]
    
    to_drop = ['Stn Name', 'CIMIS Region', 'Notes']
    df.drop(to_drop, axis=1, inplace=True)
    
    if ~(keep_dates):
        df.drop('Date', axis=1, inplace=True)
    
    # Removing one outlier
    df = df[df['Avg Wind Speed (mph)'] > 0]
    
    
    return df, name_region_map

def pipeline():
    path = '../data/conditions_df.csv'
    df, region_dict = load_and_clean(path)
    
    target = df.pop('Target')
    df_num = df.drop('Stn Id', axis=1)

    num_pipeline = Pipeline([
        ('imputer', KNNImputer(weights='distance')),
        ('scaler', StandardScaler())
    ])

    num_attribs = list(df_num.columns)
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
    return df

if __name__=="__main__":
    df = pipeline()
    print(df.shape)
    print(df.Target.value_counts() / len(df))
    print(df.head())