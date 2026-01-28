import pandas as pd
import numpy as np
import logging
import os
import json
import yaml
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ensuring "logs" exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


# setting logger
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

# console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# file logger
log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# setting formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)



# loading trained model
def load_model(model_path: str):
    """Loading trained model"""
    try:
        # loading model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Trained model loaded from: %s', model_path)
        return model
    except FileNotFoundError as e:
        logger.error('FileNotFoundError file not present in the drectory')
        raise
    except Exception as e:
        logger.error('Error occured during model loading: %s', e)
        raise


# load data from the data folder
def load_data(file_path: str) -> pd.DataFrame:
    """Load test data from the folder"""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Test data loaded completely from: %s', file_path)
        return df
    except FileNotFoundError as e:
        logger.debug('FileNotFoundError occured, test file is not present')
        raise
    except Exception as e:
        logger.error('Error occured during test data loading')
        raise

# evaluation of trained model on test data
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluation of the model and return evaluation matrices"""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        classfic_report = classification_report(y_test, y_pred, output_dict=True)
        matric_dict = {
            'accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'classfic_report': classfic_report
        }
        logger.debug('Model evaluation completed')
        return matric_dict
    except Exception as e:
        logger.error('Error occured during model evaluation: %s', e)
        raise


# saving evaluation matrics
def save_reports(matrics: dict, file_path: str):
    """Save the evaluation matrics into a JSON file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(matrics, file, indent=4)
        logger.debug('Evaluation data saved in %s', file_path)
    except Exception as e:
        logger.error('Error occured during saving evaluation reports: %s', e)
        raise
    


def main():
    """load model, evaluate it and save evaluation data, params in json file"""
    try:
        # loading model 
        model = load_model('./models/model.pkl')
        
        # loading test data
        df = load_data('./data/engineered/test_engineered.csv')
        
        # slitting data into feature and target
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
        X_test_data = df[features]
        y_test_data = df['Survived']
        
        # evaluating the trained model on test data
        metrics = evaluate_model(model, X_test_data, y_test_data)
        
        save_reports(metrics, './reports/metrics.json')
        logger.debug('Model evaluation operation completed.')
    except Exception as e:
        logger.error('Failed to complete model evaluation: %s', e)
        raise

if __name__ == '__main__':
    main()