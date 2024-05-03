import data_utils
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import data_transformation
import model_trainer
import pandas as pd
 
class DataIngestion():
    def __init__(self,                 
                 root):
        self.file_path = os.path.join(root, 'dataset', 'stud.csv')
        self.root = root
    def initiate_data_ingestion(self):
        #load dataset
        df = data_utils.load_dataset(self.file_path)
        
        #split the dataset into train and test
        train_dataset, test_dataset = train_test_split(df, 
                                                 test_size= 0.2,
                                                 random_state=42)

        #save train, test and raw data in artifacts folder
        raw_path, train_dataset_path, test_dataset_path = data_utils.DataIngestionConfiguration(self.root)
        df.to_csv(raw_path)
        train_dataset.to_csv(train_dataset_path)
        test_dataset.to_csv(test_dataset_path)
        
        return train_dataset_path, test_dataset_path
   
root = Path(__file__).parents[1]
       
obj = DataIngestion(root = root)
train_dataset_path, test_dataset_path = obj.initiate_data_ingestion()


#data trasnformation
transform_obj = data_transformation.DataTransformation(root = root,
                                                       train_path = train_dataset_path,
                                                       test_path = test_dataset_path)
train_dataset, test_dataset = transform_obj.initiate_data_transformation()

#model trainer
trainer = model_trainer.ModelTrainer()
r2_square = trainer.initiate_model_trainer(train_dataset = train_dataset, 
                                         test_dataset = test_dataset,
                                         root = root)
