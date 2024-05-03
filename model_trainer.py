from sklearn.ensemble import (
                            AdaBoostRegressor,
                            GradientBoostingRegressor,
                            RandomForestRegressor,
                            )
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import utils
import data.data_utils as data_utils
from sklearn.metrics import r2_score

class ModelTrainer():
    def __init__(self, ):
        pass
    def initiate_model_trainer(self, 
                               train_dataset,
                               test_dataset, 
                               root):
        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
            }
        train_X, test_X, train_Y, test_Y = (train_dataset.iloc[:,:-1],
                                            test_dataset.iloc[:,:-1], 
                                            train_dataset.iloc[:,-1],
                                            test_dataset.iloc[:,-1])

        model_scores = utils.evaluate_model(models=models,
                                             train_features = train_X, 
                                             train_labels = train_Y, 
                                             test_features = test_X, 
                                             test_labels = test_Y)
        #getting the best score
        best_model_name = max(model_scores, key=lambda x: model_scores[x])
        best_score = model_scores[best_model_name]
        
        print('results',model_scores)  
        print("Best Model:", best_model_name)
        print("Best Score:", best_score)
        
        best_model = models[best_model_name]
        #save the best model
        data_utils.save_object(root = root, 
                               obj_name = best_model_name)
        
        predicted = best_model.predict(test_X)
        r2_square = r2_score(test_Y, predicted)
        print(r2_square)
        return r2_square
       