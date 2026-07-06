"""
config.py
----------
Single source of truth for every path and constant used across the project.

Why this matters for the portfolio narrative:
- No hardcoded paths scattered across components/pipelines (a common beginner-notebook smell).
- Change a folder name once here, and every stage of the pipeline picks it up.
- Makes the project trivially portable between local dev, Docker, and Streamlit Cloud.

Every component/pipeline file should import paths from here rather than
building its own os.path.join calls.
"""

import os

# ---------------------------------------------------------------------------
# Project root (this file lives at <root>/src/config.py, so parent-of-parent
# is the project root regardless of where the process is launched from).
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "student_performance.csv")
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "test.csv")

# ---------------------------------------------------------------------------
# Artifact paths (models, preprocessing objects, evaluation reports)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
PREPROCESSORS_DIR = os.path.join(ARTIFACTS_DIR, "preprocessors")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")

REGRESSOR_MODEL_PATH = os.path.join(MODELS_DIR, "score_regressor.pkl")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "risk_classifier.pkl")
PREPROCESSOR_PATH = os.path.join(PREPROCESSORS_DIR, "preprocessor.pkl")

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.path.join(ROOT_DIR, "mlruns")  # local file-store backend
MLFLOW_EXPERIMENT_NAME = "student-success-intelligence"

# ---------------------------------------------------------------------------
# AI / RAG
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = os.path.join(ROOT_DIR, "ai", "rag", "knowledge_base")
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "ai", "rag", "vector_store")

# ---------------------------------------------------------------------------
# Modeling constants
# ---------------------------------------------------------------------------
TARGET_SCORE_COLUMN = "final_score"
TARGET_RISK_COLUMN = "risk_category"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Risk category thresholds derived from predicted score.
RISK_THRESHOLDS = {
    "low_risk_min": 75,     # score >= 75  -> Low Risk
    "medium_risk_min": 50,  # 50 <= score < 75 -> Medium Risk
    # score < 50 -> High Risk
}


def score_to_risk_category(score: float) -> str:
    """Single, reusable rule for converting a numeric score into a risk label.

    Keeping this logic here (instead of duplicating it in data_transformation.py,
    model_trainer.py, and the prediction_pipeline.py) guarantees training-time
    and inference-time labels can never drift apart.
    """
    if score >= RISK_THRESHOLDS["low_risk_min"]:
        return "Low Risk"
    elif score >= RISK_THRESHOLDS["medium_risk_min"]:
        return "Medium Risk"
    else:
        return "High Risk"


# ---------------------------------------------------------------------------
# Ensure required directories exist at import time (idempotent, safe to repeat).
# ---------------------------------------------------------------------------
for _dir in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREPROCESSORS_DIR,
    REPORTS_DIR,
    KNOWLEDGE_BASE_DIR,
    VECTOR_STORE_DIR,
]:
    os.makedirs(_dir, exist_ok=True)
