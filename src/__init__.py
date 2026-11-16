"""
Bank Fraud Detection - A modular fraud detection system for credit card transactions.

This package provides:
- Data loading and preprocessing
- Model training with multiple algorithms
- Comprehensive evaluation with visualizations
- Production-ready prediction interface
"""

__version__ = "1.0.0"
__author__ = "Bank Fraud Detection Team"

from src.config import *
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import FraudPredictor

__all__ = [
    "DataLoader",
    "DataPreprocessor", 
    "ModelTrainer",
    "ModelEvaluator",
    "FraudPredictor",
]