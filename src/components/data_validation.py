import os
import json
import pandas as pd

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.config import REPORTS_DIR


logger = get_logger(__name__)


class DataValidation:
    def __init__(self):
        self.report_path = os.path.join(REPORTS_DIR, "validation_report.json")

    def validate_data(self, data_path):
        try:
            logger.info("Data validation started")

            df = pd.read_csv(data_path)

            report = {
                "total_rows": df.shape[0],
                "total_columns": df.shape[1],
                "missing_values": df.isnull().sum().to_dict(),
                "duplicates": int(df.duplicated().sum()),
                "columns": list(df.columns),
                "status": "PASSED",
            }

            os.makedirs(REPORTS_DIR, exist_ok=True)

            with open(self.report_path, "w") as file:
                json.dump(report, file, indent=4)

            logger.info("Data validation completed")

            return self.report_path

        except Exception as e:
            raise CustomException(e)
