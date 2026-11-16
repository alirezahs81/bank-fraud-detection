"""
Unit tests for Bank Fraud Detection pipeline.
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config import (
    TARGET_COLUMN, DROP_COLUMNS, TEST_SIZE, RANDOM_STATE,
    MODELS_CONFIG, METRICS, DEFAULT_THRESHOLD
)
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import FraudPredictor, load_sample_transaction


class TestConfig:
    """Test configuration values."""
    
    def test_target_column(self):
        assert TARGET_COLUMN == "Class"
    
    def test_drop_columns(self):
        assert "Time" in DROP_COLUMNS
    
    def test_test_size(self):
        assert TEST_SIZE == 0.2
    
    def test_random_state(self):
        assert RANDOM_STATE == 42
    
    def test_models_config(self):
        assert "logistic_regression" in MODELS_CONFIG
        assert "random_forest" in MODELS_CONFIG
        assert "xgboost" in MODELS_CONFIG
        assert "lightgbm" in MODELS_CONFIG
    
    def test_metrics(self):
        expected_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
        assert METRICS == expected_metrics
    
    def test_default_threshold(self):
        assert DEFAULT_THRESHOLD == 0.5


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_init_default_path(self):
        loader = DataLoader()
        assert loader.data_path.name == "creditcard.csv"
    
    def test_init_custom_path(self):
        loader = DataLoader("custom/path.csv")
        # Path uses OS-specific separator
        assert loader.data_path == Path("custom/path.csv")
    
    @patch('src.data_loader.pd.read_csv')
    def test_load_data_success(self, mock_read_csv):
        mock_df = pd.DataFrame({'V1': [1, 2], 'V2': [3, 4], 'Class': [0, 1]})
        mock_read_csv.return_value = mock_df
        
        # Mock the file existence check
        with patch('src.data_loader.Path.exists', return_value=True):
            loader = DataLoader("fake_path.csv")
            df = loader.load_data()
        
        assert df.shape == (2, 3)
        mock_read_csv.assert_called_once()
    
    def test_load_data_file_not_found(self):
        loader = DataLoader("nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    def test_get_data_info(self):
        df = pd.DataFrame({
            'V1': [1, 2, 3],
            'V2': [4, 5, 6],
            'Class': [0, 0, 1]
        })
        loader = DataLoader()
        info = loader.get_data_info(df)
        
        assert info['shape'] == (3, 3)
        assert 'V1' in info['columns']
        assert info['class_distribution'] == {0: 2, 1: 1}
        assert info['fraud_percentage'] == pytest.approx(33.33, rel=0.1)


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def setup_method(self):
        # Create sample data similar to creditcard dataset with enough fraud samples for SMOTE
        np.random.seed(42)
        n_samples = 1000
        # Use more balanced data for testing (at least 6 fraud samples for SMOTE)
        self.df = pd.DataFrame({
            'Time': np.arange(n_samples),
            **{f'V{i}': np.random.randn(n_samples) for i in range(1, 29)},
            'Amount': np.random.exponential(100, n_samples),
            'Class': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])  # ~10 fraud samples
        })
        self.preprocessor = DataPreprocessor()
    
    def test_preprocess_pipeline_shape(self):
        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.preprocess_pipeline(self.df)
        
        # Check shapes - SMOTE increases train size, so total > original
        assert X_test.shape[0] == pytest.approx(len(self.df) * TEST_SIZE, rel=0.1)
        assert X_train.shape[0] >= len(self.df) * (1 - TEST_SIZE)  # At least original train size
        assert len(feature_names) == 29  # V1-V28 + Amount (Time dropped)
        assert 'Time' not in feature_names
        assert 'Amount' in feature_names
    
    def test_preprocess_pipeline_stratification(self):
        X_train, X_test, y_train, y_test, _ = self.preprocessor.preprocess_pipeline(self.df)
        
        # Check stratification maintained similar class ratios
        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()
        original_fraud_rate = self.df['Class'].mean()
        
        # After SMOTE, train should be balanced (~0.5)
        # Test should be close to original
        assert test_fraud_rate == pytest.approx(original_fraud_rate, rel=0.2)
    
    def test_drop_columns(self):
        df_with_time = self.df.copy()
        df_dropped = self.preprocessor._drop_columns(df_with_time)
        assert 'Time' not in df_dropped.columns
        assert 'Amount' in df_dropped.columns
    
    def test_separate_features_target(self):
        X, y = self.preprocessor._separate_features_target(self.df.drop(columns=['Time']))
        assert 'Class' not in X.columns
        assert y.name == 'Class'
        assert len(X) == len(y)
    
    def test_transform_new_data(self):
        # First fit the preprocessor
        self.preprocessor.preprocess_pipeline(self.df)
        
        # Create new sample
        new_sample = pd.DataFrame([{
            **{f'V{i}': 0.5 for i in range(1, 29)},
            'Amount': 100.0
        }])
        
        transformed = self.preprocessor.transform_new_data(new_sample)
        assert transformed.shape == (1, 29)


class TestModelTrainer:
    """Test ModelTrainer class."""
    
    def setup_method(self):
        # Small balanced dataset for quick testing
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.random.choice([0, 1], 200, p=[0.5, 0.5])
        
        self.X_train = X[:150]
        self.X_test = X[150:]
        self.y_train = y[:150]
        self.y_test = y[150:]
        self.trainer = ModelTrainer()
    
    def test_initialize_models(self):
        self.trainer._initialize_models(fraud_ratio=1.0)
        
        assert "logistic_regression" in self.trainer.models
        assert "random_forest" in self.trainer.models
        # XGBoost and LightGBM may not be available in test env
    
    def test_cross_validate_models(self):
        self.trainer._initialize_models(fraud_ratio=1.0)
        self.trainer._cross_validate_models(self.X_train, self.y_train)
        
        assert len(self.trainer.cv_results) > 0
        for name, results in self.trainer.cv_results.items():
            assert 'f1_mean' in results
            assert 'f1_std' in results
            assert 0 <= results['f1_mean'] <= 1
    
    def test_select_best_model(self):
        self.trainer._initialize_models(fraud_ratio=1.0)
        self.trainer._cross_validate_models(self.X_train, self.y_train)
        self.trainer._select_best_model()
        
        assert self.trainer.best_model is not None
        assert self.trainer.best_model_name in self.trainer.models
        assert 0 <= self.trainer.best_score <= 1
    
    def test_get_cv_results_summary(self):
        self.trainer._initialize_models(fraud_ratio=1.0)
        self.trainer._cross_validate_models(self.X_train, self.y_train)
        self.trainer._select_best_model()
        
        summary = self.trainer.get_cv_results_summary()
        assert isinstance(summary, pd.DataFrame)
        assert 'Model' in summary.columns
        assert 'F1_Mean' in summary.columns
        assert len(summary) == len(self.trainer.cv_results)


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    def setup_method(self):
        # Create a simple mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        self.mock_model.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.8, 0.2]
        ])
        
        self.y_test = np.array([0, 1, 0, 1, 1])  # Last one is FN
        self.evaluator = ModelEvaluator(model=self.mock_model, threshold=0.5)
    
    def test_evaluate(self):
        X_test = np.random.randn(5, 10)
        metrics = self.evaluator.evaluate(X_test, self.y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'average_precision' in metrics
        
        for v in metrics.values():
            assert 0 <= v <= 1
    
    def test_get_metrics(self):
        self.evaluator.evaluate(np.random.randn(5, 10), self.y_test)
        metrics = self.evaluator.get_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 6
    
    def test_print_evaluation_report(self):
        # Should not raise
        y_pred = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.3, 0.7, 0.2])
        ModelEvaluator.print_evaluation_report(self.y_test, y_pred, y_prob)
    
    def test_find_optimal_threshold(self):
        from src.evaluate import find_optimal_threshold
        y_test = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        optimal = find_optimal_threshold(y_test, y_prob, metric='f1')
        assert 0 < optimal < 1
        
        optimal_prec = find_optimal_threshold(y_test, y_prob, metric='precision')
        assert 0 < optimal_prec < 1


class TestFraudPredictor:
    """Test FraudPredictor class."""
    
    def setup_method(self):
        # Create mock model and scaler
        self.mock_model = MagicMock()
        # Return probabilities for single sample (most tests use single sample)
        self.mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        
        self.mock_scaler = MagicMock()
        self.mock_scaler.transform.return_value = np.array([[0.5] * 29])
        self.mock_scaler.feature_names_in_ = np.array([f'V{i}' for i in range(1, 29)] + ['Amount'])
        
        # We'll test with patching joblib.load
        self.patcher_model = patch('src.predict.joblib.load', side_effect=[self.mock_model, self.mock_scaler])
        self.patcher_model.start()
    
    def teardown_method(self):
        self.patcher_model.stop()
    
    def test_init_loads_artifacts(self):
        predictor = FraudPredictor()
        assert predictor.model == self.mock_model
        assert predictor.scaler == self.mock_scaler
        assert len(predictor.feature_names) == 29
    
    def test_predict_single_list(self):
        predictor = FraudPredictor()
        sample = [0.5] * 29
        pred = predictor.predict(sample)
        
        assert pred.shape == (1,)
        assert pred[0] in [0, 1]
    
    def test_predict_with_probability(self):
        predictor = FraudPredictor()
        sample = [0.5] * 29
        prob = predictor.predict(sample, return_probability=True)
        
        assert prob.shape == (1,)
        assert 0 <= prob[0] <= 1
    
    def test_predict_with_details(self):
        predictor = FraudPredictor()
        sample = [0.5] * 29
        details = predictor.predict_with_details(sample)
        
        assert 'prediction' in details
        assert 'label' in details
        assert 'fraud_probability' in details
        assert 'threshold' in details
        assert 'confidence' in details
        assert details['label'] in ['Legitimate', 'Fraud']
        assert 0 <= details['fraud_probability'] <= 1
        assert 0 <= details['confidence'] <= 1
    
    def test_predict_batch(self):
        # Update mock for batch (2 samples)
        self.mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        self.mock_scaler.transform.return_value = np.array([[0.5] * 29, [0.3] * 29])
        
        predictor = FraudPredictor()
        df = pd.DataFrame({f'V{i}': [0.5, 0.3] for i in range(1, 29)})
        df['Amount'] = [100, 200]
        
        result = predictor.predict_batch(df)
        
        assert 'fraud_probability' in result.columns
        assert 'fraud_prediction' in result.columns
        assert 'prediction_label' in result.columns
        assert len(result) == 2
    
    def test_set_threshold(self):
        predictor = FraudPredictor()
        predictor.set_threshold(0.3)
        assert predictor.threshold == 0.3


class TestSampleTransactions:
    """Test sample transaction loading."""
    
    def test_load_sample_transaction(self):
        samples = load_sample_transaction()
        
        assert 'legitimate' in samples
        assert 'fraud' in samples
        assert len(samples['legitimate']) == 29
        assert len(samples['fraud']) == 29
        assert isinstance(samples['legitimate'], list)
        assert isinstance(samples['fraud'], list)


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.slow
    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked data."""
        # This would be a more comprehensive test
        # For now, just verify imports work
        from src import (
            DataLoader, DataPreprocessor, ModelTrainer,
            ModelEvaluator, FraudPredictor
        )
        assert all([
            DataLoader, DataPreprocessor, ModelTrainer,
            ModelEvaluator, FraudPredictor
        ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])