from modules import constants
import pickle
import os

def rescale_target(y):
    with open(os.path.join(os.path.join(constants.model_path, 'y_scaler.pickle')), 'rb') as f:
        scaler = pickle.load(f) 
        return scaler.inverse_transform(y)
    