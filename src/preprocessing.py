"""
Data preprocessing module for Bank Fraud Detection project.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
from src.config import (
    TARGET_COLUMN, DROP_COLUMNS, TEST_SIZE, RANDOM_STATE,
    USE_SMOTE, SMOTE_RANDOM_STATE, SCALER_FILE, MODEL_DIR
)


class DataPreprocessor:
    """Handles data preprocessing including scaling, splitting, and class balancing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def preprocess_pipeline(self, df: pd.DataFrame):
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        print("Starting preprocessing pipeline...")
        
        # Step 1: Drop unnecessary columns
        df_processed = self._drop_columns(df)
        
        # Step 2: Separate features and target
        X, y = self._separate_features_target(df_processed)
        
        # Step 3: Train-test split
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Keep raw features here. Scaling and resampling are fitted inside the
        # estimator pipeline, independently within each CV training fold.
        self.is_fitted = False
        print("Preprocessing pipeline completed (raw features; scaling is deferred).")
        return X_train, X_test, y_train, y_test, self.feature_names
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop specified columns."""
        cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
        if cols_to_drop:
            print(f"Dropping columns: {cols_to_drop}")
            return df.drop(columns=cols_to_drop)
        return df
    
    def _separate_features_target(self, df: pd.DataFrame):
        """Separate features (X) and target (y)."""
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")
        
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        self.feature_names = X.columns.tolist()
        
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        print(f"Fraud cases in target: {y.sum()} ({y.mean()*100:.4f}%)")
        
        return X, y
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series):
        """Split data into train and test sets with stratification."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Train fraud rate: {y_train.mean()*100:.4f}%")
        print(f"Test fraud rate: {y_test.mean()*100:.4f}%")
        
        return X_train, X_test, y_train, y_test
    
    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Scale features using StandardScaler."""
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler (ensure the models directory exists first)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, SCALER_FILE)
        print(f"Scaler saved to {SCALER_FILE}")
        
        return X_train_scaled, X_test_scaled
    
    def _apply_smote(self, X_train: np.ndarray, y_train: pd.Series):
        """Apply SMOTE to handle class imbalance."""
        print("Applying SMOTE for class balancing...")
        print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
        
        # Check if we have enough minority samples for SMOTE
        minority_count = (y_train == 1).sum()
        if minority_count < 6:  # Default k_neighbors=5, need at least 6
            print(f"Warning: Only {minority_count} fraud samples. Skipping SMOTE (need >=6).")
            return X_train, y_train
        
        # Adjust k_neighbors based on minority class size
        k_neighbors = min(5, minority_count - 1)
        
        smote = SMOTE(random_state=SMOTE_RANDOM_STATE, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE - Class distribution: {np.bincount(y_resampled)}")
        print(f"Resampled shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def transform_new_data(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted scaler.
        
        Args:
            X: New data DataFrame.
            
        Returns:
            Scaled numpy array.
        """
        # Scaling is owned by the fitted model Pipeline. This helper now only
        # aligns raw input columns; it deliberately does not fit or load a scaler.
        if self.feature_names is None:
            raise ValueError("Feature names are unavailable; preprocess training data first")
        return X[self.feature_names].to_numpy()
    
    def load_scaler(self):
        """Load the saved scaler."""
        self.scaler = joblib.load(SCALER_FILE)
        self.is_fitted = True