from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import data_utils

class DataTransformation():
    def __init__(self, root):
        self.root = root
    def get_data_transformer_object(self):
        #specify categorical and numerical features
        num_feat= ['reading_score','writing_score']
        cat_feat = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
        
        #specify data transformation pipeline/sequence
        num_pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')),
                                         ('scaler',StandardScaler())
                                         ]
                                )
        cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")),
                                 ('onehotencoding',OneHotEncoder() ),
                                 ('scaler', StandardScaler(with_mean=False))])
        
        preprocessor = ColumnTransformer([('num_preprocess', num_pipeline, num_feat),
                                             ('cat_preprocess', cat_pipeline, cat_feat)
                                             ]
                                            )
        return preprocessor
    
    def initiate_data_transformation(self,
                                     train_data,
                                     test_data):
        preprocessor = self.get_data_transformer_object()
        
        #specify input and target data
        target_column = 'math_score'
        train_X = train_data.drop(target_column, axis = 1)
        train_Y = train_data[target_column]
        test_X = test_data.drop(target_column, axis = 1)
        test_Y = test_data[target_column]
        
        #Applying data transformation
        train_X_array = preprocessor.fit_transform(train_X)
        test_X_array = preprocessor.transform(test_X)
        
        #convert transformed array to a dataframe
        train_X_scaled = pd.DataFrame(train_X_array)
        test_X_scaled = pd.DataFrame(test_X_array)
        train_dataset = pd.concat([train_X_scaled, train_Y], axis = 1)
        test_dataset = pd.concat([test_X_scaled, test_Y], axis = 1)
        data_utils.save_object(root = self.root,  
                               obj_name= 'preprocessor.pkl' )
        
        return train_dataset, test_dataset
        