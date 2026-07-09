from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.rag_advisor import RAGAdvisor


if __name__ == "__main__":
    # ================================
    # STEP 1: DATA INGESTION
    # ================================

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    print("Data ingestion completed")

    # ================================
    # STEP 2: DATA VALIDATION
    # ================================

    validator = DataValidation()

    validation_report = validator.validate_data(train_path)

    print("Data validation completed")

    print(validation_report)

    # ================================
    # STEP 3: DATA TRANSFORMATION
    # ================================

    transformer = DataTransformation()

    (X_train, X_test, y_train, y_test) = transformer.transform_data(
        train_path, test_path
    )

    print("Feature engineering completed")

    print("Training shape:", X_train.shape)

    print("Testing shape:", X_test.shape)

    # ================================
    # STEP 4: MODEL TRAINING
    # ================================

    trainer = ModelTrainer()

    score = trainer.train(X_train, X_test, y_train, y_test)

    print("Model Training Completed")

    print(f"Model R2 Score: {score}")

    # ================================
    # STEP 5: AI RAG ADVISOR
    # ================================

    advisor = RAGAdvisor()

    advice = advisor.generate_advice(
        prediction=score,
        weak_features=[
            "math score",
            "test preparation course",
            "previous academic performance",
        ],
    )

    print("\n========== AI STUDENT ADVISOR ==========\n")

    print(advice)

    print("\nComplete AI Student Success Pipeline Finished Successfully 🚀")
