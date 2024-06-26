from sklearn.metrics import r2_score
import pickle
import os

def evaluate_model(models, 
                   train_features, 
                   train_labels, 
                   test_features,
                   test_labels):
    results = {}
    
    best_model_score = float('-inf')
    for model_name, model in models.items():
        
        #train the model
        model.fit(train_features, train_labels)
        train_pred_labels = model.predict(train_features)
        test_pred_labels = model.predict(test_features)
        
        #calculate R-squared of regression model
        train_model_score = r2_score(train_labels, train_pred_labels)
        test_model_score = r2_score(test_labels, test_pred_labels)
        # print("R-squared using r2_score function:", train_model_score)
        results.update({model_name:test_model_score})     
           
    return results


def save_object(root, obj, obj_name):
    obj_path = os.path.join(root, 'data', 'artifacts', obj_name)
    with open(obj_path, 'wb') as file:
        return pickle.dump(obj, file)
    
def load_object(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)