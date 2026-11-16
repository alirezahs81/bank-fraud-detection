"""
Prediction module for Bank Fraud Detection project.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.config import MODEL_FILE, SCALER_FILE, TARGET_COLUMN, DROP_COLUMNS, DEFAULT_THRESHOLD
from src.preprocessing import DataPreprocessor


class FraudPredictor:
    """Handles fraud predictions for new transactions."""
    
    def __init__(self, model_path=None, scaler_path=None, threshold=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model.
            scaler_path: Path to saved scaler.
            threshold: Classification threshold.
        """
        self.model_path = Path(model_path) if model_path else MODEL_FILE
        self.scaler_path = Path(scaler_path) if scaler_path else SCALER_FILE
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and scaler."""
        print(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        
        print(f"Loading scaler from {self.scaler_path}...")
        self.scaler = joblib.load(self.scaler_path)
        
        # Get feature names from scaler
        if hasattr(self.scaler, 'feature_names_in_'):
            self.feature_names = self.scaler.feature_names_in_.tolist()
        else:
            # Fallback: expect 29 features (V1-V28 + Amount)
            self.feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        print(f"Predictor ready. Features: {len(self.feature_names)}")
    
    def predict(self, transaction_data, return_probability=False):
        """
        Predict fraud for a single transaction or batch.
        
        Args:
            transaction_data: List, numpy array, or DataFrame with transaction features.
            return_probability: If True, return fraud probability instead of binary prediction.
            
        Returns:
            Prediction result (0=Legitimate, 1=Fraud) or probability.
        """
        # Convert input to DataFrame
        if isinstance(transaction_data, (list, np.ndarray)):
            transaction_data = np.array(transaction_data)
            if transaction_data.ndim == 1:
                transaction_data = transaction_data.reshape(1, -1)
            df = pd.DataFrame(transaction_data, columns=self.feature_names)
        elif isinstance(transaction_data, pd.DataFrame):
            df = transaction_data.copy()
        else:
            raise ValueError("Input must be list, numpy array, or DataFrame")
        
        # Preprocess
        df_processed = self._preprocess(df)
        
        # Scale
        X_scaled = self.scaler.transform(df_processed)
        
        # Predict
        if return_probability:
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            return probabilities
        else:
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)
            return predictions
    
    def predict_with_details(self, transaction_data):
        """
        Predict with detailed output.
        
        Args:
            transaction_data: Single transaction as list, array, or dict.
            
        Returns:
            Dictionary with prediction details.
        """
        prob = self.predict(transaction_data, return_probability=True)
        pred = (prob >= self.threshold).astype(int)
        
        result = {
            'prediction': int(pred[0]),
            'label': 'Fraud' if pred[0] == 1 else 'Legitimate',
            'fraud_probability': float(prob[0]),
            'threshold': self.threshold,
            'confidence': float(max(prob[0], 1 - prob[0]))
        }
        
        return result
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data to match training format."""
        # Drop columns that were dropped during training
        cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        return df
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for a batch of transactions.
        
        Args:
            df: DataFrame with transaction data.
            
        Returns:
            DataFrame with predictions added.
        """
        df = df.copy()
        probabilities = self.predict(df, return_probability=True)
        predictions = (probabilities >= self.threshold).astype(int)
        
        df['fraud_probability'] = probabilities
        df['fraud_prediction'] = predictions
        df['prediction_label'] = df['fraud_prediction'].map({0: 'Legitimate', 1: 'Fraud'})
        
        return df
    
    def set_threshold(self, threshold: float):
        """Update classification threshold."""
        self.threshold = threshold
        print(f"Threshold updated to {threshold}")


def load_sample_transaction():
    """Load a sample transaction for testing (from the dataset)."""
    # This is a sample legitimate transaction (first row typically)
    sample_legitimate = [
        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388,
        0.239599, 0.098698, 0.363787, 0.090794, -0.551600, -0.617801,
        -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791,
        0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928,
        0.128539, -0.189115, 0.133558, -0.021053, 149.62
    ]
    
    # Sample fraud transaction (characteristics vary)
    sample_fraud = [
        1.191857, 0.266151, 0.166480, 0.448154, 0.060018, -0.082361,
        -0.078803, 0.085102, -0.255425, -0.166974, 1.612727, 1.065235,
        0.489095, -0.143772, 0.635558, 0.463917, -0.114804, -0.183361,
        -0.145783, -0.069083, -0.225784, -0.638672, 0.101288, -0.339846,
        0.167170, 0.125895, -0.008983, 0.014724, 2125.87
    ]
    
    return {
        'legitimate': sample_legitimate,
        'fraud': sample_fraud
    }