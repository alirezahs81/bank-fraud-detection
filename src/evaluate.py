"""
Model evaluation module for Bank Fraud Detection project.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
)
import joblib
from pathlib import Path
from src.config import MODEL_FILE, TARGET_COLUMN


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations."""
    
    def __init__(self, model=None, threshold=0.5):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model (if None, loads from disk).
            threshold: Classification threshold for probability predictions.
        """
        self.model = model
        self.threshold = threshold
        self.y_test = None
        self.y_pred = None
        self.y_prob = None
        
        if model is None:
            self.load_model()
    
    def load_model(self):
        """Load model from disk."""
        self.model = joblib.load(MODEL_FILE)
        print(f"Model loaded from {MODEL_FILE}")
    
    def evaluate(self, X_test, y_test, threshold=None):
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features.
            y_test: True labels.
            threshold: Classification threshold (overrides default).
        """
        self.y_test = y_test
        self.threshold = threshold if threshold is not None else self.threshold
        
        # Get predictions
        self.y_prob = self.model.predict_proba(X_test)[:, 1]
        self.y_pred = (self.y_prob >= self.threshold).astype(int)
        
        return self.get_metrics()
    
    def get_metrics(self) -> dict:
        """Calculate all evaluation metrics."""
        if self.y_test is None or self.y_pred is None:
            raise ValueError("Must call evaluate() first")
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1': f1_score(self.y_test, self.y_pred),
        }
        
        if self.y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, self.y_prob)
            metrics['average_precision'] = average_precision_score(self.y_test, self.y_prob)
        
        return metrics
    
    @staticmethod
    def print_evaluation_report(y_test, y_pred, y_prob=None):
        """Print formatted evaluation report."""
        print("\n=== Model Evaluation Report ===")
        print(f"Accuracy:     {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision:    {precision_score(y_test, y_pred):.4f}")
        print(f"Recall:       {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score:     {f1_score(y_test, y_pred):.4f}")
        
        if y_prob is not None:
            print(f"ROC-AUC:      {roc_auc_score(y_test, y_prob):.4f}")
            print(f"Avg Precision: {average_precision_score(y_test, y_prob):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Legitimate', 'Fraud']
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'Confusion Matrix (Threshold={self.threshold})')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve."""
        if self.y_prob is None:
            raise ValueError("Probabilities not available. Call evaluate() first.")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(self.y_test, self.y_prob, ax=ax)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_title(f'ROC Curve (AUC = {roc_auc_score(self.y_test, self.y_prob):.4f})')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, save_path=None):
        """Plot Precision-Recall curve."""
        if self.y_prob is None:
            raise ValueError("Probabilities not available. Call evaluate() first.")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(self.y_test, self.y_prob, ax=ax)
        
        # Baseline (random classifier)
        baseline = self.y_test.mean()
        ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline (AP={baseline:.4f})')
        
        ax.set_title(f'Precision-Recall Curve (AP = {average_precision_score(self.y_test, self.y_prob):.4f})')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        
        return fig
    
    def plot_threshold_analysis(self, save_path=None):
        """Plot metrics vs threshold to find optimal threshold."""
        if self.y_prob is None:
            raise ValueError("Probabilities not available. Call evaluate() first.")
        
        thresholds = np.arange(0.01, 1.0, 0.01)
        precisions = []
        recalls = []
        f1_scores = []
        
        for t in thresholds:
            y_pred_t = (self.y_prob >= t).astype(int)
            precisions.append(precision_score(self.y_test, y_pred_t, zero_division=0))
            recalls.append(recall_score(self.y_test, y_pred_t, zero_division=0))
            f1_scores.append(f1_score(self.y_test, y_pred_t, zero_division=0))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Precision-Recall vs Threshold
        axes[0].plot(thresholds, precisions, label='Precision', color='blue')
        axes[0].plot(thresholds, recalls, label='Recall', color='red')
        axes[0].plot(thresholds, f1_scores, label='F1 Score', color='green')
        axes[0].axvline(x=self.threshold, color='black', linestyle='--', label=f'Current ({self.threshold})')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Metrics vs Threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall curve
        PrecisionRecallDisplay.from_predictions(self.y_test, self.y_prob, ax=axes[1])
        axes[1].set_title('Precision-Recall Curve')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis saved to {save_path}")
        
        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        print(f"\nOptimal threshold (max F1): {optimal_threshold:.3f}")
        print(f"  Precision: {precisions[optimal_idx]:.4f}")
        print(f"  Recall: {recalls[optimal_idx]:.4f}")
        print(f"  F1: {f1_scores[optimal_idx]:.4f}")
        
        return fig, optimal_threshold
    
    def plot_all(self, save_dir=None):
        """Generate all evaluation plots."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_confusion_matrix(save_dir / "confusion_matrix.png" if save_dir else None)
        self.plot_roc_curve(save_dir / "roc_curve.png" if save_dir else None)
        self.plot_precision_recall_curve(save_dir / "pr_curve.png" if save_dir else None)
        self.plot_threshold_analysis(save_dir / "threshold_analysis.png" if save_dir else None)
        
        plt.show()


def find_optimal_threshold(y_test, y_prob, metric='f1'):
    """
    Find optimal threshold based on specified metric.
    
    Args:
        y_test: True labels.
        y_prob: Predicted probabilities.
        metric: Metric to optimize ('f1', 'precision', 'recall').
        
    Returns:
        Optimal threshold value.
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    scores = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        if metric == 'f1':
            scores.append(f1_score(y_test, y_pred_t, zero_division=0))
        elif metric == 'precision':
            scores.append(precision_score(y_test, y_pred_t, zero_division=0))
        elif metric == 'recall':
            scores.append(recall_score(y_test, y_pred_t, zero_division=0))
    
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx]