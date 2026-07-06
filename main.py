from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


if __name__ == "__main__":


    ingestion = DataIngestion()


    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )


    validation = DataValidation()


    report = validation.validate_data(
        train_path
    )


    print(
        "Phase 2 pipeline completed"
    )


    print(
        report
    )