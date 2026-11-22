"""
Main entry point for Bank Fraud Detection pipeline.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader, download_kaggle_dataset
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import FraudPredictor, load_sample_transaction
from src.config import MODEL_FILE


def main():
    print("=== Starting Bank Fraud Detection Pipeline ===")
    
    # Step 1: Load Data
    print("\n[Step 1/5] Loading Data...")
    loader = DataLoader()
    
    try:
        df = loader.load_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nDownloading instructions:")
        download_kaggle_dataset()
        return
    
    loader.print_data_info(df)
    
    # Step 2: Preprocess Data
    print("\n[Step 2/5] Preprocessing Data (scaling inside model CV pipeline)...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(df)
    print(f"Preprocessing completed. Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Step 3: Train Models
    print("\n[Step 3/5] Training Models & Selecting Best Estimator...")
    trainer = ModelTrainer()
    best_model = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Print CV results summary
    print("\n=== Cross-Validation Results Summary ===")
    cv_summary = trainer.get_cv_results_summary()
    print(cv_summary.to_string(index=False))
    
    # Step 4: Comprehensive Evaluation
    print("\n[Step 4/5] Comprehensive Evaluation...")
    evaluator = ModelEvaluator(best_model)
    evaluator.evaluate(X_test, y_test)
    evaluator.print_evaluation_report(y_test, evaluator.y_pred, evaluator.y_prob)
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    evaluator.plot_all()
    
    # Step 5: Demo Predictions
    print("\n[Step 5/5] Running Sample Predictions...")
    predictor = FraudPredictor()
    samples = load_sample_transaction()
    
    print("\n--- Sample Legitimate Transaction ---")
    result_legit = predictor.predict_with_details(samples['legitimate'])
    print(f"Prediction: {result_legit['label']}")
    print(f"Fraud Probability: {result_legit['fraud_probability']:.6f}")
    print(f"Confidence: {result_legit['confidence']:.4f}")
    
    print("\n--- Sample Fraud Transaction ---")
    result_fraud = predictor.predict_with_details(samples['fraud'])
    print(f"Prediction: {result_fraud['label']}")
    print(f"Fraud Probability: {result_fraud['fraud_probability']:.6f}")
    print(f"Confidence: {result_fraud['confidence']:.4f}")
    
    print("\n=== Pipeline Completed Successfully! ===")
    print(f"\nBest model saved to: {MODEL_FILE}")
    print("\nTo make predictions on new data:")
    print("  from src.predict import FraudPredictor")
    print("  predictor = FraudPredictor()")
    print("  result = predictor.predict_with_details(your_transaction_data)")


if __name__ == "__main__":
    main()