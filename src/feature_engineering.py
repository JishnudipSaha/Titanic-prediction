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
        df.fillna(inplace=True)
        logger.debug('Data Loaded Successfully from %s', data_url)
        return df
    except Exception as e:
        logger.error('Error occured during data loading: %s', e)
        raise

def main():
    """
    Docstring for main
    """
    train_data = load_data('./data/interim/train_processed.csv')
    