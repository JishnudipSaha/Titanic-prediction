import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# Ensure "logs" directory exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# setting console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# setting file handler
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)


# defining and setting formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handling object
logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(data_url: str) -> pd.DataFrame:
    """Loading data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded successfully from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Preprocess the data"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug('Train and Test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occured while saving data: %s', e)
        raise

def main():
    try:
        test_size = 0.2
        data_path = "https://raw.githubusercontent.com/JishnudipSaha/Datasets/refs/heads/main/Titanic-Dataset.csv"
        df = load_data(data_url=data_path)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data=train_data, test_data=test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete data ingestion process.')

if __name__ == '__main__':
    main()