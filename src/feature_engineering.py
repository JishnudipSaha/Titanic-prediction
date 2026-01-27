import pandas as pd
import logging
import os

# Ensuring "log" exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# setting logger for both console console and file
logger = logging.getLogger('feature_engieering')
logger.setLevel(logging.DEBUG)

# setting condole handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# setting file handler
log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# setting formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handler to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load train and test data from ./data/interim
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data Loaded Successfully from %s', data_url)
        return df
    except Exception as e:
        logger.error('Error occured during data loading: %s', e)
        raise

def engineer_df(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Engineering preprocessed data"""
    try:
        # nothing much to feature engineer, so pass it on as it is.
        logger.debug('Preprocessed data have been engineered.')
        return train_df, test_df
    except Exception as e:
        logger.error('Error occured during engineering preprocessed data: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Savig egineered data as a csv file."""
    try:
        engineer_data_path = os.path.join(data_path, 'engineered')
        os.makedirs(engineer_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(engineer_data_path, 'train_engineered.csv'), index=False)
        test_data.to_csv(os.path.join(engineer_data_path, 'test_engineered.csv'), index=False)
        logger.debug('Engineered data file save to %s', engineer_data_path)
    except Exception as e:
        logger.error('Error occured during saving data %s', e)
        raise

def main():
    """
    Docstring for main
    """
    try:
        # loading data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        # featuring engineering preprocessed data
        train_engr_data, test_engr_data = engineer_df(train_data, test_data)
        
        # saving engineered data
        save_data(train_engr_data, test_engr_data, './data')
        logger.debug('Full feature engineering operation completed.')
    except Exception as e:
        logger.error('Failed to complete feature engineering operation.')
        raise

if __name__ == '__main__':
    main()