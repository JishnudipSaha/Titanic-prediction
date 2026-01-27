import pandas as pd
import numpy as np
import logging
import os

# ensure "log" folder exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# setting logger for both console and file logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

# for console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# for file logger
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# setting formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_df(dataFrame: pd.DataFrame) -> pd.DataFrame:
    """
    This will preprocess data by:
    Age: Filling with median value
    Embarked: fill with most common value (mode)
    Encoding catagorical column:
    Sex: male: 0, female: 1
    Embarked: one-hot encoding (drop first to avoid multicollinearity)
    """
    try:
        logger.debug('Starting preprocessing of the DataFrame')
        # Age: filling with median values
        dataFrame['Age'].fillna(dataFrame['Age'].median(), inplace=True)
        # Embarked: fill with most common value (mode)
        dataFrame['Embarked'].fillna(dataFrame['Embarked'].mode()[0], inplace=True)
        
        # Sex: Encoding catagorical data
        dataFrame['Sex'] = dataFrame['Sex'].map({'male': 0, 'female': 1})
        # Embarked: one-hot encoding (drop first to avoid multicollinearity)
        dataFrame = pd.get_dummies(dataFrame, columns=['Embarked'], drop_first=True)
        
        logger.debug('Data preprocessing successfully completed.')
        return dataFrame
    
    except Exception as e:
        logger.error('Error occured during data preprocessing %s', e)
        raise
        

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Saves both preprocessed train and test data into new file.
    """
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(interim_data_path, 'train_processed.csv'), index=False)
        test_data.to_csv(os.path.join(interim_data_path, 'test_processed.csv'), index=False)
        logger.debug('Preprocessed data saves successfully to %s', interim_data_path)
    except Exception as e:
        logger.error('Error during saving preprocessed data: %s', e)
        raise
    
def main():
    """
    Main function to load raw data and preprocess it and save the preprocess data
    """
    try:
        # loading raw data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        
        # preprocessing data
        train_preprocessed_data = preprocess_df(train_data)
        test_preprocessed_data = preprocess_df(test_data)
        
        # saving the preprocessed data
        save_data(train_preprocessed_data, test_preprocessed_data, './data')
        logger.debug('Preprocess operation completed successfully')
    except Exception as e:
        logger.error('Preprocessing operation failed: %s', e)


if __name__ == '__main__':
    main()