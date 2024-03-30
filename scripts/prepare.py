import sys
sys.path.insert(0, '..')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from modules import constants
import pandas as pd
import pickle
import os
    
    
if __name__ == '__main__':
    print('Reading data...')
    train_metadata = pd.read_csv(constants.train_metadata_path)
    X_test = pd.read_csv(constants.test_metadata_path)
    # define columns
    all_cols = train_metadata.columns
    world_climate_vars = list(all_cols[all_cols.str.startswith('WORLDCLIM')])
    soil_vars = list(all_cols[all_cols.str.startswith('SOIL')])
    modis_vars = list(all_cols[all_cols.str.startswith('MODIS')])
    vod_vars = list(all_cols[all_cols.str.startswith('VOD')])
    train_vars = ['id'] + world_climate_vars + soil_vars + modis_vars + vod_vars 
    target_vars = list(all_cols[all_cols.str.endswith('_mean')])
    # dataset split
    print('Splitting data...')
    X = train_metadata[train_vars]
    y = train_metadata[target_vars]
    X_test = X_test[train_vars]
    X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.2, random_state=42)

    # scaler definition 
    print('Preparing data...')
    X_transformer = ColumnTransformer([
        ("standarize", StandardScaler(), train_vars[1:])
    ], remainder='passthrough')
    y_transformer = ColumnTransformer([
        ("standarize", StandardScaler(), target_vars)
    ])
    X_transformer.set_output(transform = 'pandas')
    y_transformer.set_output(transform = 'pandas')
    X_transformer.fit(X_train)
    y_transformer.fit(y_train)
    print('Saving scalers...')
    with open(os.path.join(constants.model_path, 'X_transformer.pickle'), 'wb') as f:
        pickle.dump(X_transformer, f)
    with open(os.path.join(constants.model_path, 'y_transformer.pickle'), 'wb') as f:
        pickle.dump(y_transformer, f)
    
    # transformations
    X_train = pd.DataFrame(X_transformer.transform(X_train))
    X_val = pd.DataFrame(X_transformer.transform(X_val))
    X_test = pd.DataFrame(X_transformer.transform(X_test))
    y_train = pd.DataFrame(y_transformer.transform(y_train))
    y_val = pd.DataFrame(y_transformer.transform(y_val))

    # saving
    print('Saving...')
    X_train.to_feather(os.path.join(constants.prepared_data_path, 'X_train.arrow'))
    X_val.to_feather(os.path.join(constants.prepared_data_path, 'X_val.arrow'))
    X_test.to_feather(os.path.join(constants.prepared_data_path, 'X_test.arrow'))
    y_train.to_feather(os.path.join(constants.prepared_data_path, 'y_train.arrow'))
    y_val.to_feather(os.path.join(constants.prepared_data_path, 'y_val.arrow'))