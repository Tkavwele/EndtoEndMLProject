import pandas as pd
import utils
import os
from pathlib import Path

class predictPipeline():
    def __init__(self):
        
        pass
    def predict(self, features): 
        root = Path(__file__).parents[1] 
        print(root)
        model_path = os.path.join( root, 'data', 'artifacts', 'model.pkl')
        preprocessor_path = os.path.join( root, 'data', 'artifacts', 'preprocessor.pkl')
        model = utils.load_object(file_path = model_path)
        preprocessor = utils.load_object(file_path = preprocessor_path)
        
        scaled_data = preprocessor.transform(features)
        predictions = model.predict(scaled_data)

        return predictions
    
class CustomData():
    def __init__(self,
                 gender: str,
                race_ethnicity: str,
                parental_level_of_education: str,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score
        
    def get_data_as_data_frame(self):
        custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
        return pd.DataFrame(custom_data_input_dict)

# obj = predictPipeline()
# obj1 = CustomData(gender = 'female',
# race_ethnicity = 'group B',
# parental_level_of_education = "bachelor's degree",
# lunch = 'standard',
# test_preparation_course = 'none',
# reading_score =66,
# writing_score = 39)
# pred_df = obj1.get_data_as_data_frame()
# predictions = obj.predict(pred_df)
