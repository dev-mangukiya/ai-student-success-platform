import os
import pandas as pd

from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.config import (
    RAW_DATA_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    PROCESSED_DATA_DIR,
)


logger = get_logger(__name__)


class DataIngestion:
    def __init__(self):
        self.raw_path = RAW_DATA_FILE

        self.train_path = TRAIN_DATA_FILE

        self.test_path = TEST_DATA_FILE

    def initiate_data_ingestion(self):
        try:
            logger.info("Data ingestion started")

            df = pd.read_csv(self.raw_path)

            logger.info(f"Dataset loaded successfully: {df.shape}")

            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

            train, test = train_test_split(df, test_size=0.2, random_state=42)

            train.to_csv(self.train_path, index=False)

            test.to_csv(self.test_path, index=False)

            logger.info("Train test split completed")

            return (self.train_path, self.test_path)

        except Exception as e:
            raise CustomException(e)
