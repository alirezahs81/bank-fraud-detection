"""
Model training module for Bank Fraud Detection project.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.config import USE_SMOTE, SMOTE_RANDOM_STATE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
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
        
        # Fit the complete pipeline on the untouched training split. The
        # scaler and SMOTE are fitted only on each training fold / final train set.
        self.best_model.fit(X_train, y_train)
        
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
    
    def _build_pipeline(self, estimator):
        """Build leakage-safe preprocessing, resampling and estimator pipeline."""
        steps = [("scaler", StandardScaler())]
        if USE_SMOTE:
            steps.append(("smote", SMOTE(random_state=SMOTE_RANDOM_STATE, k_neighbors=5)))
        steps.append(("classifier", estimator))
        return ImbPipeline(steps=steps)

    def _cross_validate_models(self, X_train, y_train):
        """Evaluate each model with scaling and SMOTE fitted within CV folds."""
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "accuracy": "accuracy", "precision": "precision", "recall": "recall",
            "f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision"
        }
        print(f"\\nPerforming {CV_FOLDS}-fold cross-validation...")
        for name, estimator in self.models.items():
            print(f"\\nEvaluating {name}...")
            pipeline = self._build_pipeline(estimator)
            result = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=scoring,
                n_jobs=-1, return_train_score=False, error_score="raise"
            )
            detailed = {
                metric: {"mean": float(np.mean(result[f"test_{metric}"])),
                         "std": float(np.std(result[f"test_{metric}"])),
                         "scores": result[f"test_{metric}"]}
                for metric in scoring
            }
            f1_scores = result["test_f1"]
            self.cv_results[name] = {
                "f1_scores": f1_scores, "f1_mean": float(np.mean(f1_scores)),
                "f1_std": float(np.std(f1_scores)), "detailed": detailed
            }
            print(f"  F1 Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores)*2:.4f})")
            print(f"  Precision: {detailed['precision']['mean']:.4f}")
            print(f"  Recall: {detailed['recall']['mean']:.4f}")
            print(f"  ROC-AUC: {detailed['roc_auc']['mean']:.4f}")
            print(f"  Avg Precision: {detailed['average_precision']['mean']:.4f}")

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
        self.best_model = self._build_pipeline(self.models[self.best_model_name])
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
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
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