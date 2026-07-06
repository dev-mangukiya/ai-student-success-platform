import os

from sklearn.metrics import r2_score

from xgboost import XGBRegressor


from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.helpers import save_object


logger = get_logger(__name__)



class ModelTrainer:


    def __init__(self):

        self.model_path = (
            "artifacts/models/model.pkl"
        )



    def train(

        self,

        X_train,

        X_test,

        y_train,

        y_test

    ):


        try:


            logger.info(
                "Model training started"
            )



            model = XGBRegressor(

                n_estimators=200,

                learning_rate=0.05,

                max_depth=5,

                random_state=42

            )



            model.fit(

                X_train,

                y_train

            )



            predictions = model.predict(

                X_test

            )



            score = r2_score(

                y_test,

                predictions

            )



            logger.info(

                f"Model R2 score: {score}"

            )



            os.makedirs(

                "artifacts/models",

                exist_ok=True

            )



            save_object(

                self.model_path,

                model

            )



            logger.info(

                "Model saved successfully"

            )



            return score



        except Exception as e:


            raise CustomException(e)