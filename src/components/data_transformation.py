import os

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.helpers import save_object
from src.config import PREPROCESSOR_PATH, PREPROCESSORS_DIR


logger = get_logger(__name__)


class DataTransformation:
    def __init__(self):
        self.preprocessor_path = PREPROCESSOR_PATH

    def get_preprocessor(self):
        numerical_features = ["reading score", "writing score"]

        categorical_features = [
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course",
        ]

        numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_pipeline = Pipeline(
            steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", numeric_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

        return preprocessor

    def transform_data(self, train_path, test_path):
        try:
            logger.info("Transformation started")

            train = pd.read_csv(train_path)

            test = pd.read_csv(test_path)

            target = "math score"

            X_train = train.drop(target, axis=1)

            y_train = train[target]

            X_test = test.drop(target, axis=1)

            y_test = test[target]

            preprocessor = self.get_preprocessor()

            X_train = preprocessor.fit_transform(X_train)

            X_test = preprocessor.transform(X_test)

            os.makedirs(PREPROCESSORS_DIR, exist_ok=True)

            save_object(self.preprocessor_path, preprocessor)

            logger.info("Transformation completed")

            return (X_train, X_test, y_train, y_test)

        except Exception as e:
            raise CustomException(e)
