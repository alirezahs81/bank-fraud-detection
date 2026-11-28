# Bank Fraud Detection System

[![CI](https://github.com/alirezahs81/bank-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/bank-fraud-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Production-oriented **credit-card fraud detection** on the [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset — leakage-safe training, multi-model selection, evaluation, and inference.

**Monitoring (MLOps):** pair this repo with [**fraud-model-monitor**](https://github.com/alirezahs81/fraud-model-monitor) for prediction logging, PSI/KS drift, and delayed-label metrics. See [docs/MONITORING.md](docs/MONITORING.md).

---

## Key features

| Feature | Description |
|----------|---------|
| **Smart preprocessing** | Fold-local feature scaling and SMOTE inside a leakage-safe pipeline |
| **Multiple algorithms** | Logistic Regression, Random Forest, XGBoost, LightGBM |
| **Cross-validation** | Best model selected by F1 with Stratified K-Fold |
| **Evaluation** | Confusion matrix, ROC, PR curve, threshold analysis |
| **Inference** | `FraudPredictor` for single/batch transactions |
| **Monitoring handoff** | Export `best_fraud_model.joblib` → fraud-model-monitor |

---

## Project structure

```text
bank-fraud-detection/
├── data/raw/creditcard.csv
├── models/best_fraud_model.joblib
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── main.py
├── docs/MONITORING.md
├── tests/
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/alirezahs81/bank-fraud-detection.git
cd bank-fraud-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Dataset:** place `creditcard.csv` under `data/raw/` (Kaggle download or API).

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

---

## Train

```bash
python -m src.main
```

1. Load data  
2. Preprocess + split  
3. CV training / model selection  
4. Evaluation plots  
5. Prediction demo  

Artifact: `models/best_fraud_model.joblib`

---

## Inference

```python
from src.predict import FraudPredictor

predictor = FraudPredictor()
new_transaction = [
    -1.359807, -0.072781, 2.536347, 1.378155, -0.338321,
    0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
    -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
    -0.470401, 0.207971, 0.025791, 0.403993, 0.251412,
    -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
    -0.189115, 0.133558, -0.021053, 149.62
]
print(predictor.predict_with_details(new_transaction))
```

---

## Register for monitoring

```bash
# from fraud-model-monitor repo:
python -m src.register_model \
  --model /path/to/bank-fraud-detection/models/best_fraud_model.joblib
uvicorn src.api:app --port 8000
```

More detail: [docs/MONITORING.md](docs/MONITORING.md)

---

## Tests

```bash
pytest tests/ -v
```

---

## Notes

1. Class imbalance (~0.17% fraud): SMOTE only inside CV folds; test split never resampled.  
2. Features: V1–V28 (PCA) + Amount; `Time` dropped by default.  
3. Benchmark numbers in older notes should be re-run after pipeline changes before citing them.

---

## License

MIT License — free to use, modify and distribute.
