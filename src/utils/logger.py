"""
logger.py
----------
Centralized logging utility for the AI Student Success Intelligence Platform.

Design goals:
- One logger configuration used by every module (components, pipelines, ai/rag, dashboard).
- Logs written to a timestamped file under `logs/` AND streamed to console.
- Easy to import: `from src.utils.logger import get_logger` then `logger = get_logger(__name__)`.

This mirrors how production ML systems (and most solid MLOps portfolio projects)
implement logging: one place to configure format/handlers, everywhere else just
asks for a logger by name.
"""

import logging
import os
from datetime import datetime

# Directory where all run logs are stored.
# Kept at project root /logs so it doesn't pollute src/.
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# One log file per process run, timestamped so historical runs are never overwritten.
_LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, _LOG_FILE_NAME)

_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"

# Configure the root logger exactly once per process.
#
# NOTE: We deliberately do NOT guard this with `if not _root_logger.handlers`.
# Test runners like pytest (and some notebook kernels) attach their own
# handlers to the root logger before this module is ever imported, which
# would make that guard always false and silently skip creating our file
# handler. Instead we tag our own handler with a custom attribute and check
# for THAT specifically, so re-imports stay idempotent without being fooled
# by handlers other tools added.
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)

_already_configured = any(
    getattr(h, "_student_success_ai_handler", False) for h in _root_logger.handlers
)

if not _already_configured:
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_handler._student_success_ai_handler = True  # tag for the idempotency check above

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    console_handler._student_success_ai_handler = True

    _root_logger.addHandler(file_handler)
    _root_logger.addHandler(console_handler)


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Return a named logger that inherits the root file+console configuration.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Data ingestion started")
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    # Quick manual test: `python -m src.utils.logger`
    logger = get_logger(__name__)
    logger.info("Logger initialized successfully.")
    logger.warning("This is a warning-level test log.")
    logger.error("This is an error-level test log.")
    print(f"Log file written to: {LOG_FILE_PATH}")
