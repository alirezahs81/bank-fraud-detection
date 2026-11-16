"""
Model training module for Bank Fraud Detection project.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.config import USE_SMOTE, SMOTE_RANDOM_STATE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, make_scorer
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try importing optional models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.config import MODELS_CONFIG, MODEL_FILE, CV_FOLDS, RANDOM_STATE, METRICS


class ModelTrainer:
    """Handles model training, cross-validation, and model selection."""
    
    def __init__(self):
        self.models = {}
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train all models and select the best one based on cross-validation.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Best trained model.
        """
        print("\n=== Model Training & Evaluation ===")
        
        # Calculate scale_pos_weight for XGBoost
        fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Initialize models
        self._initialize_models(fraud_ratio)
        
        # Cross-validation for each model (SMOTE applied inside each fold)
        self._cross_validate_models(X_train, y_train)
        
        # Select best model
        self._select_best_model()
        
        # Final fit on the SMOTE-resampled training split
        X_train_fit, y_train_fit = X_train, y_train
        if USE_SMOTE:
            X_train_fit, y_train_fit = self._resample_train(X_train, y_train)
        self.best_model.fit(X_train_fit, y_train_fit)
        
        # Final evaluation on test set
        self._evaluate_on_test_set(X_test, y_test)
        
        # Save best model
        self._save_best_model()
        
        return self.best_model
    
    def _initialize_models(self, fraud_ratio):
        """Initialize all models with their configurations."""
        for name, config in MODELS_CONFIG.items():
            # Skip unavailable models
            if name == "xgboost" and not XGBOOST_AVAILABLE:
                print(f"Skipping {name}: XGBoost not installed")
                continue
            if name == "lightgbm" and not LIGHTGBM_AVAILABLE:
                print(f"Skipping {name}: LightGBM not installed")
                continue
            
            # Adjust params for XGBoost
            params = config["params"].copy()
            if name == "xgboost":
                params["scale_pos_weight"] = fraud_ratio
            
            # Create model instance
            model_class = globals()[config["class"]]
            self.models[name] = model_class(**params)
            print(f"Initialized {name}")
    
    def _cross_validate_models(self, X_train, y_train):
        """Perform cross-validation for all models."""
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'
        }
        
        print(f"\nPerforming {CV_FOLDS}-fold cross-validation...")
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            cv_results = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='f1', n_jobs=-1
            )
            
            # Also get detailed scores for all metrics
            detailed_scores = {}
            for metric_name, metric_scorer in scoring.items():
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric_scorer, n_jobs=-1)
                detailed_scores[metric_name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
            
            self.cv_results[name] = {
                'f1_scores': cv_results,
                'f1_mean': cv_results.mean(),
                'f1_std': cv_results.std(),
                'detailed': detailed_scores
            }
            
            print(f"  F1 Score: {cv_results.mean():.4f} (+/- {cv_results.std()*2:.4f})")
            print(f"  Precision: {detailed_scores['precision']['mean']:.4f}")
            print(f"  Recall: {detailed_scores['recall']['mean']:.4f}")
            print(f"  ROC-AUC: {detailed_scores['roc_auc']['mean']:.4f}")
            print(f"  Avg Precision: {detailed_scores['average_precision']['mean']:.4f}")
    
    def _resample_train(self, X_train, y_train):
        """Resample the training split with SMOTE for the final model fit."""
        print("Resampling training split with SMOTE for final fit...")
        minority_count = int((y_train == 1).sum())
        k_neighbors = min(5, minority_count - 1)
        smote = SMOTE(random_state=SMOTE_RANDOM_STATE, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Class distribution: {np.bincount(y_res)}")
        return X_res, y_res

    def _select_best_model(self):
        """Select the best model based on cross-validation F1 score."""
        best_f1 = 0
        for name, results in self.cv_results.items():
            if results['f1_mean'] > best_f1:
                best_f1 = results['f1_mean']
                self.best_model_name = name
                self.best_score = best_f1
        
        # Train best model on full training data
        self.best_model = self.models[self.best_model_name]
        print(f"\nBest model: {self.best_model_name} (CV F1: {self.best_score:.4f})")
    
    def _evaluate_on_test_set(self, X_test, y_test):
        """Evaluate the best model on the test set."""
        print(f"\n=== Test Set Evaluation: {self.best_model_name} ===")
        
        # Fit on full training data (will be done in main after this)
        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1] if hasattr(self.best_model, "predict_proba") else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            metrics['average_precision'] = average_precision_score(y_test, y_prob)
        
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        self.test_metrics = metrics
    
    def _save_best_model(self):
        """Save the best model to disk."""
        joblib.dump(self.best_model, MODEL_FILE)
        print(f"\nBest model saved to {MODEL_FILE}")
    
    def get_cv_results_summary(self) -> pd.DataFrame:
        """Get a summary DataFrame of cross-validation results."""
        rows = []
        for name, results in self.cv_results.items():
            row = {'Model': name, 'F1_Mean': results['f1_mean'], 'F1_Std': results['f1_std']}
            for metric, scores in results['detailed'].items():
                row[f'{metric}_mean'] = scores['mean']
                row[f'{metric}_std'] = scores['std']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('F1_Mean', ascending=False).reset_index(drop=True)
        return df