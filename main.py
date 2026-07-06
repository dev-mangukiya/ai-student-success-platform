"""
main.py
--------
Single CLI entry point for the AI Student Success Intelligence Platform.

Right now (Phase 1) this just verifies the environment, logger, and custom
exception handling are wired correctly. From Phase 3 onward, this will call:

    from src.pipeline.training_pipeline import run_training_pipeline
    run_training_pipeline()

Run with:
    python main.py
"""

import sys

from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("AI Student Success Intelligence Platform — bootstrap check")
    logger.info("=" * 60)

    try:
        logger.info("Phase 1 status: project structure, logger, exceptions OK.")
        logger.info("Training pipeline will be wired in from Phase 3 onward.")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
