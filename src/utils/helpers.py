"""
helpers.py
-----------
Small, reusable utility functions shared across pipeline components.

Kept deliberately generic (no business logic here) so components/*.py stay
focused on their single responsibility (ingestion, transformation, training...).
"""

import os
import sys
import json
import dill  # more robust than pickle for saving sklearn/xgboost pipelines + lambdas

from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_object(file_path: str, obj) -> None:
    """Serialize any Python object (model, preprocessor, encoder...) to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)

        logger.info(f"Object saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Load a previously saved object back into memory."""
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)

    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path: str, data: dict) -> None:
    """Persist a dict as pretty-printed JSON (used for evaluation reports)."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, default=str)

        logger.info(f"JSON report saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_json(file_path: str) -> dict:
    """Load a JSON file back into a dict."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)

    except Exception as e:
        raise CustomException(e, sys)
