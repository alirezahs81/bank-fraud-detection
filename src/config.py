"""
Configuration settings for Bank Fraud Detection project.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATA_FILE = RAW_DATA_DIR / "creditcard.csv"

# Model paths
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "best_fraud_model.joblib"
SCALER_FILE = MODEL_DIR / "scaler.joblib"

# Model training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target column
TARGET_COLUMN = "Class"

# Features to drop (if any)
DROP_COLUMNS = ["Time"]  # Time column usually not useful for fraud detection

# Class imbalance handling
USE_SMOTE = True
SMOTE_RANDOM_STATE = 42

# Models to try
MODELS_CONFIG = {
    "logistic_regression": {
        "class": "LogisticRegression",
        "params": {
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced"
        }
    },
    "random_forest": {
        "class": "RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "n_jobs": -1
        }
    },
    "xgboost": {
        "class": "XGBClassifier",
        "params": {
            "n_estimators": 100,
            "random_state": RANDOM_STATE,
            "scale_pos_weight": 1,  # Will be calculated based on class ratio
            "eval_metric": "logloss",
            "n_jobs": -1
        }
    },
    "lightgbm": {
        "class": "LGBMClassifier",
        "params": {
            "n_estimators": 100,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "n_jobs": -1,
            "verbose": -1
        }
    }
}

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

# Threshold for fraud detection (can be tuned)
DEFAULT_THRESHOLD = 0.5