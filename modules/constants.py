import os

# root path
root_path = '/home/pol/projects/plant_traits_2024'

# data paths
data_path = os.path.join(root_path, 'data') 
train_images_path = os.path.join(data_path, 'train_images')
test_images_path = os.path.join(data_path, 'test_images')
raw_data_path = os.path.join(data_path, 'raw')
prepared_data_path = os.path.join(data_path, 'prepared')

train_metadata_path = os.path.join(raw_data_path, 'train.csv')
test_metadata_path = os.path.join(raw_data_path, 'test.csv')
target_name_metadata_path = os.path.join(raw_data_path, 'target_name_meta.tsv')
sample_submission_path = os.path.join(raw_data_path, 'sample_submission.csv')

# models paths
model_path = os.path.join(root_path, 'models')