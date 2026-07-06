"""
test_phase1_setup.py
----------------------
Smoke tests for Phase 1: project setup, config, logger, custom exceptions.

Run with:
    pytest tests/test_phase1_setup.py -v
"""

import os
import sys

import pytest

from src import config
from src.utils.logger import get_logger, LOG_FILE_PATH
from src.utils.exception import CustomException
from src.utils.helpers import save_object, load_object


def test_config_paths_exist():
    """All directories declared in config.py should exist after import."""
    assert os.path.isdir(config.RAW_DATA_DIR)
    assert os.path.isdir(config.PROCESSED_DATA_DIR)
    assert os.path.isdir(config.MODELS_DIR)
    assert os.path.isdir(config.PREPROCESSORS_DIR)
    assert os.path.isdir(config.REPORTS_DIR)


def test_score_to_risk_category_mapping():
    """Verify the score -> risk label rule matches the spec exactly."""
    assert config.score_to_risk_category(90) == "Low Risk"
    assert config.score_to_risk_category(75) == "Low Risk"
    assert config.score_to_risk_category(74.9) == "Medium Risk"
    assert config.score_to_risk_category(50) == "Medium Risk"
    assert config.score_to_risk_category(49.9) == "High Risk"
    assert config.score_to_risk_category(10) == "High Risk"


def test_logger_writes_to_file():
    """Logger should write to a real log file on disk."""
    logger = get_logger(__name__)
    logger.info("Phase 1 pytest smoke test log line")
    assert os.path.exists(LOG_FILE_PATH)
    with open(LOG_FILE_PATH, "r") as f:
        contents = f.read()
    assert "Phase 1 pytest smoke test log line" in contents


def test_custom_exception_message_contains_context():
    """CustomException should embed file name, line number, and message."""
    try:
        _ = 1 / 0
    except Exception as e:
        with pytest.raises(CustomException) as exc_info:
            raise CustomException(e, sys)
        message = str(exc_info.value)
        assert "division by zero" in message
        assert "line number" in message


def test_save_and_load_object_roundtrip(tmp_path):
    """save_object/load_object should round-trip an arbitrary Python object."""
    sample_obj = {"model": "xgboost", "version": 1}
    file_path = os.path.join(tmp_path, "sample.pkl")

    save_object(file_path, sample_obj)
    assert os.path.exists(file_path)

    loaded = load_object(file_path)
    assert loaded == sample_obj
