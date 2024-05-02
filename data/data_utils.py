import pandas as pd
import os
import pickle

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def DataIngestionConfiguration(root):
    folder = os.path.join(root,'data', 'artifacts')
    raw_path = os.path.join(folder, 'data.csv' )
    train_data_path = os.path.join(folder, 'train_data.csv' )
    test_data_path = os.path.join(folder, 'test_data.csv' )
    
    os.makedirs(folder, exist_ok=True)
    
    return raw_path, train_data_path, test_data_path
def save_object(root, obj_name):
    obj_path = os.path.join(root, 'data', 'artifacts', obj_name)
    with open(obj_path, 'wb') as file:
        return pickle.dump(obj_name, file)