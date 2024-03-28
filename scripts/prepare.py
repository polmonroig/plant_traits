import sys
sys.path.insert(0, '..')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    train_vars = world_climate_vars + soil_vars + modis_vars + vod_vars 
    target_vars = list(all_cols[all_cols.str.endswith('_mean')])
    # dataset split
    print('Splitting data...')
    X = train_metadata[train_vars]
    y = train_metadata[target_vars]
    X_test = X_test[train_vars]
    X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    # scaler definition 
    print('Preparing data...')
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaler.fit(X_train)
    y_scaler.fit(y_train)
    print('Saving scalers...')
    with open(os.path.join(constants.model_path, 'X_scaler.pickle'), 'wb') as f:
        pickle.dump(X_scaler, f)
    with open(os.path.join(constants.model_path, 'y_scaler.pickle'), 'wb') as f:
        pickle.dump(y_scaler, f)
    
    # transformations
    X_train = pd.DataFrame(X_scaler.transform(X_train))
    X_val = pd.DataFrame(X_scaler.transform(X_val))
    X_test = pd.DataFrame(X_scaler.transform(X_test))
    y_train = pd.DataFrame(y_scaler.transform(y_train))
    y_val = pd.DataFrame(y_scaler.transform(y_val))

    # saving
    print('Saving...')
    X_train.to_feather(os.path.join(constants.prepared_data_path, 'X_train.arrow'))
    X_val.to_feather(os.path.join(constants.prepared_data_path, 'X_val.arrow'))
    X_test.to_feather(os.path.join(constants.prepared_data_path, 'X_test.arrow'))
    y_train.to_feather(os.path.join(constants.prepared_data_path, 'y_train.arrow'))
    y_val.to_feather(os.path.join(constants.prepared_data_path, 'y_val.arrow'))