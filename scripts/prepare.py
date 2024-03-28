from sklearn.preprocessing import StandardScaler
from modules import constants
import pandas as pd
import pickle
import os

def standarize_data(df, name):
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    with open(os.path.join(constants.model_path, name)) as f:
        pickle.dump(scaler, f)

    return df
    

def prepare_metadata(df, kind):
    """
    Send dataframe to transform and kind (train/test)
    """
    all_cols = train_metadata.columns
    world_climate_vars = list(all_cols[all_cols.str.startswith('WORLDCLIM')])
    soil_vars = list(all_cols[all_cols.str.startswith('SOIL')])
    modis_vars = list(all_cols[all_cols.str.startswith('MODIS')])
    vod_vars = list(all_cols[all_cols.str.startswith('VOD')])
    train_vars = world_climate_vars + soil_vars + modis_vars + vod_vars 
    X = df[train_vars]
    X = standarize_data(X, 'X_' + kind + '_scaler.pickle')
    # save objects 
    X.to_feather(os.path.join(constants.prepared_data_path, 'X_' + kind + '.arrow'))
    if kind == 'train': 
        target_vars = list(all_cols[all_cols.str.endswith('_mean')])
        y = df[target_vars]
        y = standarize_data(y, 'y_' + kind + '_scaler.pickle')
        y.to_feather(os.path.join(constants.prepared_data_path, 'y_' + kind + '.arrow'))
    


    
if __name__ == '__main__':

    train_metadata = pd.read_csv(constants.train_metadata_path)
    test_metadata = pd.read_csv(constants.test_metadata_path)
    prepare_metadata(train_metadata, 'train')
    prepare_metadata(test_metadata, 'test')