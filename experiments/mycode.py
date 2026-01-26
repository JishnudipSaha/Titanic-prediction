import pandas as pd
import os
import logging


# logger configuration
logger = logging.getLogger('mycode')
logger.setLevel('DEBUG')

# using logger at a terminal/console level
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')
# formatting the logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# adding log handler
logger.addHandler(console_handler)



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

def save_data(df: pd.DataFrame, data_path: str) -> None:
    """Saving csv data to a local folder."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        df.to_csv(os.path.join(raw_data_path, "data.csv"), index=False)
        logger.debug('Data successfully saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occur while saving the data: %s', e)
        raise

def main():
    try:
        data_path = "https://raw.githubusercontent.com/JishnudipSaha/Datasets/refs/heads/main/Titanic-Dataset.csv"
        df = load_data(data_url=data_path)
        save_data(df=df, data_path='./experiments/data') # this will make a data folder in root
        logger.debug('Data loading and saving successfully completed')
    except Exception as e:
        logger.error('Failed to complete data loading and saving: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()