# Production monitoring

After training, serve scores and track drift with the sibling project
[**fraud-model-monitor**](https://github.com/alirezahs81/fraud-model-monitor).

## Contract

| Item | Value |
|------|--------|
| Artifact | `models/best_fraud_model.joblib` (full sklearn/imblearn pipeline preferred) |
| Features | `V1` … `V28`, `Amount` (no `Time`) |
| API | `predict_proba` → column 1 = fraud probability |

## Register in the monitor

```bash
# 1) Train here
python -m src.main

# 2) Register artifact for monitoring
cd ../fraud-model-monitor   # or clone that repo
python -m src.register_model \
  --model /absolute/path/to/bank-fraud-detection/models/best_fraud_model.joblib

# 3) Run monitor API
uvicorn src.api:app --port 8000
```

Details: [fraud-model-monitor/docs/INTEGRATION_BANK_FRAUD.md](https://github.com/alirezahs81/fraud-model-monitor/blob/main/docs/INTEGRATION_BANK_FRAUD.md)
