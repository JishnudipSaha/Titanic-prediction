import pandas as pd
import logging
import os
from sklearn.linear_model import LogisticRegression
import pickle
import yaml

# Ensure "logs" exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# method to load params from params.yaml
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


# method to load data from a path
def load_data(data_path: str) -> pd.DataFrame:
    """Load CSV data from the given location."""
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data set loaded from %s', data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Error during data loading from "%s" error: %s', data_path, e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured during data loading %s', e)
        raise

# method to train model from training dataset
def train_model(train_df: pd.DataFrame, params: dict) -> LogisticRegression:
    """Model training: Logistic Regression"""
    try:
        # model building on training data
        logger.debug('Started training ML model.')
        max_iter = params['max_iter']
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
        X_train = train_df[features]
        y_train = train_df['Survived']
        lr_model = LogisticRegression(max_iter=max_iter) # max_iter = 200
        lr_model.fit(X_train, y_train) # both features from train dataset.
        logger.debug('Model training completed.')
        return lr_model
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error occured during model training: %s', e)
        raise

# method to save trained model to a .pkl file
def save_model(model: LogisticRegression, file_path: str) -> None:
    """Save mode into a model folder"""
    try:
        # ensuring the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to: %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured during model saving: %s', e)
        raise


def main():
    """Load data engineered data, train model(LogisticReg) then saving model"""
    try:
        
        # loading params from params.yaml
        params = load_params('params.yaml')['model_building']
        
        # loading model
        train_data = load_data('./data/engineered/train_engineered.csv')
        
        # train mode on train data
        model = train_model(train_data, params=params)
        
        # saving train model
        save_model(model, './models/model.pkl')
        
        logger.debug('Model training operation completed.')
    except Exception as e:
        logger.error('Model training operation failed: %s', e)
        raise

if __name__ == "__main__":
    main()