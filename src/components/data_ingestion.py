import os, sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_values_path: str = os.path.join('data', 'train_values.csv')
    raw_labels_path: str = os.path.join('data', 'train_labels.csv')

    artifactss_dir : str = 'artifacts'
    merged_data_path : str = os.path.join('artifacts', 'raw.csv')
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def inititate_data_ingestion(self):
        logging.info('Starting data ingestion')

        try:
            os.makedirs(self.ingestion_config.artifactss_dir, exist_ok=True)

            logging.info(f'Reading building values from {self.ingestion_config.raw_values_path}')
            df_values = pd.read_csv(self.ingestion_config.raw_values_path)


            logging.info(f'Reading building values from {self.ingestion_config.raw_labels_path}')
            df_labels = pd.read_csv(self.ingestion_config.raw_labels_path)

            df = pd.merge(df_values, df_labels, on='building_id', how='inner')
            logging.info(f'shape of df after merge :- {df.shape}')


            train_set, test_set= train_test_split(df, test_size=0.2, random_state=42, stratify=df['damage_grade'])
            logging.info(f'train shape {train_set.shape} and test shape {test_set.shape}')

            train_set.to_csv(self.ingestion_config.train_data_path, index = False)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False)

            logging.info('Data Ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )


        except Exception as e:
            CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.inititate_data_ingestion()
    print("Train data path", train_path)
    print("Test data path", test_path)
