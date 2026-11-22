# 🏦 Bank Fraud Detection System

[![CI](https://github.com/alirezahs81/bank-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/bank-fraud-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An advanced, modular and production-oriented Python project for **detecting fraud in credit card transactions** using machine learning.

Built on the well-known [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), covering all stages of a complete ML project: from data loading and preprocessing to model training, comprehensive evaluation and inference.

---

## 🛠️ Key Features

| Feature | Description |
|----------|---------|
| **Smart preprocessing** | Fold-local feature scaling and class-imbalance handling with SMOTE inside a single leakage-safe pipeline |
| **Multiple algorithms** | Logistic Regression, Random Forest, XGBoost, LightGBM |
| **Cross-validation** | Best model selected by F1-Score with Stratified K-Fold |
| **Comprehensive evaluation** | Confusion Matrix, ROC Curve, Precision-Recall Curve, Threshold Analysis |
| **Fast inference** | `FraudPredictor` class for real-time prediction on new transactions |
| **Modular structure** | Clean, extensible and testable code |

---

## 📁 Project Structure

```text
Bank Fraud Detection/
│
├── data/
│   └── raw/
│       └── creditcard.csv          # Main dataset (downloaded from Kaggle)
│
├── models/
│   └── best_fraud_model.joblib     # Complete fitted pipeline (scaler + classifier)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Project configuration
│   ├── data_loader.py              # Data loading & validation
│   ├── preprocessing.py            # Raw feature preparation and split
│   ├── train.py                    # Model training & selection (SMOTE inside CV)
│   ├── evaluate.py                 # Evaluation & visualization
│   ├── predict.py                  # Production inference
│   └── main.py                     # Main pipeline entry point
│
├── tests/
│   └── test_pipeline.py            # Unit tests
│
├── requirements.txt                # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/alirezahs81/bank-fraud-detection.git
cd bank-fraud-detection
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
**Automatic (requires Kaggle API):**
```bash
# Set up the Kaggle API (once)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

**Manual:**
1. Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it at `data/raw/creditcard.csv`

---

## 🚀 Running the Full Pipeline

```bash
python -m src.main
```

This runs the following steps in order:
1. **Data loading** — read CSV and display basic info
2. **Preprocessing** — scaling, train/test split (SMOTE applied per CV fold inside the trainer)
3. **Model training** — cross-validation over 4 algorithms
4. **Comprehensive evaluation** — metrics, ROC/PR/Confusion Matrix plots
5. **Prediction demo** — test on sample transactions

---

## 📊 Evaluation Outputs

The project automatically produces the following plots:

| Plot | Description |
|---------|---------|
| **Confusion Matrix** | TP, TN, FP, FN at the default 0.5 threshold |
| **ROC Curve** | Area under curve (AUC) for discriminative power |
| **Precision-Recall Curve** | Essential for highly imbalanced data (~0.17% fraud) |
| **Threshold Analysis** | Find the optimal threshold to maximize F1/Precision/Recall |

---

## 🔮 Production Inference

```python
from src.predict import FraudPredictor

# Initialization
predictor = FraudPredictor()

# A new transaction (list of 29 features: V1-V28 + Amount)
new_transaction = [
    -1.359807, -0.072781, 2.536347, 1.378155, -0.338321,
    0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
    -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
    -0.470401, 0.207971, 0.025791, 0.403993, 0.251412,
    -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
    -0.189115, 0.133558, -0.021053, 149.62
]

# Simple prediction (0 = legitimate, 1 = fraud)
prediction = predictor.predict(new_transaction)
print(f"Prediction: {prediction[0]}")  # 0 or 1

# Prediction with full details
details = predictor.predict_with_details(new_transaction)
print(details)
# {
#     'prediction': 0,
#     'label': 'Legitimate',
#     'fraud_probability': 0.000234,
#     'threshold': 0.5,
#     'confidence': 0.999766
# }

# Batch prediction on a DataFrame
import pandas as pd
df_transactions = pd.read_csv("new_transactions.csv")
results_df = predictor.predict_batch(df_transactions)
print(results_df[['fraud_prediction', 'fraud_probability', 'prediction_label']].head())
```

### Threshold Tuning
```python
# For higher sensitivity (fewer false negatives)
predictor.set_threshold(0.3)

# For higher precision (fewer false positives)
predictor.set_threshold(0.7)
```

---

## ⚙️ Key Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|---------|
| `TEST_SIZE` | 0.2 | Test data ratio |
| `USE_SMOTE` | True | Enable SMOTE for class balancing |
| `CV_FOLDS` | 5 | Number of cross-validation folds |
| `DEFAULT_THRESHOLD` | 0.5 | Default classification threshold |
| `MODELS_CONFIG` | - | Hyperparameter settings for each model |

---

## 🧪 Running the Tests

```bash
pytest tests/ -v
```

---

## 📈 Benchmark Results

On the Credit Card Fraud Detection dataset (284,807 transactions, 492 frauds ≈ 0.17%):

| Model | CV F1-Score | Test F1 | Test ROC-AUC | Test Avg Precision |
|------|-------------|---------|--------------|-------------------|
| Logistic Regression | ~0.82 | ~0.80 | ~0.97 | ~0.85 |
| Random Forest | ~0.85 | ~0.83 | ~0.98 | ~0.88 |
| **XGBoost** ⭐ | — | — | **0.977** | **0.847** |
| LightGBM | ~0.87 | ~0.85 | ~0.99 | ~0.90 |

The table and figures above are historical benchmark notes and have not yet been re-measured with the corrected fold-local pipeline. Re-run the full training pipeline before treating them as current performance claims.

> 💡 **Note**: XGBoost and LightGBM usually perform best. If they are not installed, the project works with Random Forest and Logistic Regression.

---

## 🔧 Development & Customization

### Adding a new model
Add it to `MODELS_CONFIG` in `src/config.py`:
```python
"catboost": {
    "class": "CatBoostClassifier",
    "params": {
        "iterations": 100,
        "random_state": 42,
        "class_weights": [1, 100],
        "verbose": False
    }
}
```

### Using your own data
Place your CSV in `data/raw/` and update `config.py`:
```python
TARGET_COLUMN = "your_fraud_column"
DROP_COLUMNS = ["columns_to_drop"]
```

---

## 📝 Important Notes

1. **Class ratio**: the dataset is highly imbalanced (Fraud: 0.17%). Scaling and SMOTE are fit only on each CV training fold; the final pipeline is fit on the training split only. The held-out test split is never resampled.
2. **Features**: V1–V28 are PCA components (anonymized data). Only `Amount` and `Time` are original.
3. **Time**: usually not useful for fraud detection and is dropped by default.
4. **Threshold**: 0.5 by default. Use `plot_threshold_analysis` in production to find the optimum.
5. **Fast inference**: the model and scaler are saved under `models/` and loaded without retraining.

---

## 📄 License

MIT License — free to use, modify and distribute.
