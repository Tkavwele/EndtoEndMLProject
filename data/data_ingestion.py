import data_utils
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import data_transformation
 
class DataIngestion():
    def __init__(self,                 
                 root):
        self.file_path = os.path.join(root, 'dataset', 'stud.csv')
        self.root = root
    def initiate_data_ingestion(self):
        #load dataset
        df = data_utils.load_dataset(self.file_path)
        
        #split the dataset into train and test
        train_data, test_data = train_test_split(df, 
                                                 test_size= 0.2,
                                                 random_state=42)

        #save train, test and raw data in artifacts folder
        raw_path, train_data_path, test_data_path = data_utils.DataIngestionConfiguration(self.root)
        df.to_csv(raw_path)
        train_data.to_csv(train_data_path)
        test_data.to_csv(test_data_path)
        
        return train_data, test_data
   
root = Path(__file__).parents[1]
print(root)        
obj = DataIngestion(root = root)
train_data, test_data = obj.initiate_data_ingestion()

#data trasnformation
transform_obj = data_transformation.DataTransformation(root = root)
train_dataset, test_dataset = transform_obj.initiate_data_transformation(train_data= train_data, 
                                                                         test_data = test_data)