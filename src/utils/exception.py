"""
exception.py
-------------
Custom exception handling for the AI Student Success Intelligence Platform.

Why a custom exception class?
- Default Python tracebacks are noisy and don't tell you WHICH pipeline stage failed.
- In a multi-stage ML pipeline (ingestion -> validation -> transformation -> training),
  you want every raised error to clearly report: file name, line number, and the
  original error message, wrapped consistently.

Usage pattern used throughout src/:

    from src.utils.exception import CustomException
    from src.utils.logger import get_logger
    import sys

    logger = get_logger(__name__)

    try:
        risky_operation()
    except Exception as e:
        logger.error("Data ingestion failed")
        raise CustomException(e, sys)
"""

import sys


def get_error_detail_message(error: Exception, error_detail: "sys") -> str:
    """
    Build a detailed, human-readable error message including:
    - the file in which the error occurred
    - the exact line number
    - the original exception message

    `error_detail` is expected to be the `sys` module (passed in as `sys` at the
    call site) so we can pull the live exception traceback via sys.exc_info().
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        # Raised outside an `except` block (no active traceback) — fall back gracefully.
        return f"Error: {str(error)}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in script: [{file_name}] "
        f"at line number: [{line_number}] "
        f"with message: [{str(error)}]"
    )


class CustomException(Exception):
    """
    Project-wide custom exception.

    Wrap any caught exception with this before re-raising so every failure
    surfaced in logs / MLflow / Streamlit carries consistent, debuggable context.
    """

    def __init__(self, error_message: Exception, error_detail: "sys" = sys):
        super().__init__(str(error_message))
        self.error_message = get_error_detail_message(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message


if __name__ == "__main__":
    # Quick manual test: `python -m src.utils.exception`
    try:
        _ = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)
