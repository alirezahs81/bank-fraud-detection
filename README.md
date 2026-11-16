# 🏦 Bank Fraud Detection System

یک پروژه پایتون پیشرفته، ماژولار و تولید‌محور برای **تشخیص تقلب در تراکنش‌های کارت اعتباری** با استفاده از الگوریتم‌های یادگیری ماشین.

این پروژه بر اساس دیتاست معروف **Credit Card Fraud Detection** از کگل ساخته شده و شامل تمام مراحل یک پروژه ML کامل است: از بارگذاری دیتا و پیش‌پردازش تا آموزش مدل، ارزیابی جامع و استنتاج (Inference).

---

## 🛠️ قابلیت‌های کلیدی

| ویژگی | توضیح |
|----------|---------|
| **پیش‌پردازش هوشمند** | اسکیل‌کردن ویژگی‌ها، حذف ستون‌های بی‌فنی، و مدیریت ناهمسانی کلاس‌ها با SMOTE |
| **چندین الگوریتم** | Logistic Regression، Random Forest، XGBoost، LightGBM |
| **اعتبارسنجی cross-validation** | انتخاب بهترین مدل بر اساس F1-Score با Stratified K-Fold |
| **ارزیابی جامع** | Confusion Matrix، ROC Curve، Precision-Recall Curve، تحلیل آستانه (Threshold Analysis) |
| **استنتاج سریع** | کلاس `FraudPredictor` برای پیش‌بینی بلادرنگ روی تراکنش‌های جدید |
| **ساختار ماژولار** | کد تمیز، قابل توسعه و تست‌پذیر |

---

## 📁 ساختار پروژه

```text
Bank Fraud Detection/
│
├── data/
│   └── raw/
│       └── creditcard.csv          # دیتاست اصلی (دانلود از کگل)
│
├── models/
│   ├── best_fraud_model.joblib     # بهترین مدل آموزش‌دیده
│   └── scaler.joblib               # اسکیلر برای پیش‌پردازش
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # تنظیمات پروژه
│   ├── data_loader.py              # بارگذاری و اعتبارسنجی دیتا
│   ├── preprocessing.py            # پیش‌پردازش، اسکیل، SMOTE
│   ├── train.py                    # آموزش مدل‌ها و انتخاب بهترین
│   ├── evaluate.py                 # ارزیابی و بصری‌سازی
│   ├── predict.py                  # استنتاج (Inference) برای تولید
│   └── main.py                     # نقطه ورود اصلی پایپ‌لاین
│
├── tests/
│   └── test_pipeline.py            # تست‌های واحد
│
├── requirements.txt                # وابستگی‌ها
└── README.md
```

---

## ⚙️ نصب و راه‌اندازی

### ۱. کلون پروژه و ورود به پوشه
```bash
cd "Bank Fraud Detection"
```

### ۲. ساخت و فعال‌سازی محیط مجازی
```bash
python -m venv venv
# ویندوز:
venv\Scripts\activate
# لینوکس/مک:
source venv/bin/activate
```

### ۳. نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### ۴. دانلود دیتاست
**روش خودکار (نیاز به کگل API):**
```bash
# تنظیم کگل API (یک‌بار)
# فایل kaggle.json را در ~/.kaggle/ (لینوکس/مک) یا C:\Users\<username>\.kaggle\ (ویندوز) قرار دهید

kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

**روش دستی:**
1. به [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) بروید
2. فایل `creditcard.csv` را دانلود کنید
3. در مسیر `data/raw/creditcard.csv` قرار دهید

---

## 🚀 اجرای پایپ‌لاین کامل

```bash
python -m src.main
```

این دستور مراحل زیر را به ترتیب اجرا می‌کند:
1. **بارگذاری دیتا** - خواندن CSV و نمایش اطلاعات پایه
2. **پیش‌پردازش** - اسکیل‌کردن، Train/Test Split، SMOTE
3. **آموزش مدل‌ها** - Cross-validation روی ۴ الگوریتم
4. **ارزیابی جامع** - متریک‌ها، نمودارهای ROC/PR/Confusion Matrix
5. **دمو پیش‌بینی** - تست روی تراکنش‌های نمونه

---

## 📊 خروجی‌های ارزیابی

پروژه به طور خودکار نمودارهای زیر را تولید می‌کند:

| نمودار | توضیح |
|---------|---------|
| **Confusion Matrix** | TP, TN, FP, FN با آستانه پیش‌فرض 0.5 |
| **ROC Curve** | مساحت زیر منحنی (AUC) برای تشخیص قدرت تفکیک |
| **Precision-Recall Curve** | مهم برای دیتاهای ناهمسان (Fraud ~0.17%) |
| **Threshold Analysis** | یافتن آستانه بهینه برای بهینه‌سازی F1/Precision/Recall |

---

## 🔮 استفاده در تولید (Production Inference)

```python
from src.predict import FraudPredictor

# инициализация
predictor = FraudPredictor()

# تراکنش جدید (لیست ۲۹ ویژگی: V1-V28 + Amount)
new_transaction = [
    -1.359807, -0.072781, 2.536347, 1.378155, -0.338321,
    0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
    -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
    -0.470401, 0.207971, 0.025791, 0.403993, 0.251412,
    -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
    -0.189115, 0.133558, -0.021053, 149.62
]

# پیش‌بینی ساده (0=سالم، 1=تقلب)
prediction = predictor.predict(new_transaction)
print(f"Prediction: {prediction[0]}")  # 0 یا 1

# پیش‌بینی با جزئیات کامل
details = predictor.predict_with_details(new_transaction)
print(details)
# {
#     'prediction': 0,
#     'label': 'Legitimate',
#     'fraud_probability': 0.000234,
#     'threshold': 0.5,
#     'confidence': 0.999766
# }

# پیش‌بینی دسته‌ای (Batch) روی DataFrame
import pandas as pd
df_transactions = pd.read_csv("new_transactions.csv")
results_df = predictor.predict_batch(df_transactions)
print(results_df[['fraud_prediction', 'fraud_probability', 'prediction_label']].head())
```

### تغییر آستانه (Threshold Tuning)
```python
# برای حساسیت بیشتر (کمتر False Negative)
predictor.set_threshold(0.3)

# برای دقت بیشتر (کمتر False Positive)  
predictor.set_threshold(0.7)
```

---

## ⚙️ تنظیمات مهم (config.py)

| پارامتر | پیش‌فرض | توضیح |
|-----------|---------|---------|
| `TEST_SIZE` | 0.2 | نسبت دیتای تست |
| `USE_SMOTE` | True | فعال‌سازی SMOTE برای بالانس کلاس‌ها |
| `CV_FOLDS` | 5 | تعداد فولدرهای Cross-Validation |
| `DEFAULT_THRESHOLD` | 0.5 | آستانه پیش‌فرض klasyfikیشن |
| `MODELS_CONFIG` | - | تنظیمات هایپرپارامترها برای هر مدل |

---

## 🧪 اجرای تست‌ها

```bash
pytest tests/ -v
```

---

## 📈 نتایج معمول (Benchmark)

روی دیتاست Credit Card Fraud Detection (284,807 تراکنش، 492 تقلب ~0.17%):

| مدل | CV F1-Score | Test F1 | Test ROC-AUC | Test Avg Precision |
|------|-------------|---------|--------------|-------------------|
| Logistic Regression | ~0.82 | ~0.80 | ~0.97 | ~0.85 |
| Random Forest | ~0.85 | ~0.83 | ~0.98 | ~0.88 |
| **XGBoost** | **~0.88** | **~0.86** | **~0.99** | **~0.91** |
| LightGBM | ~0.87 | ~0.85 | ~0.99 | ~0.90 |

> 💡 **نکته**: XGBoost و LightGBM معمولاً بهترین عملکرد را دارند. اگر نصب نباشند، پروژه با Random Forest و Logistic Regression کار می‌کند.

---

## 🔧 توسعه و سفارشی‌سازی

### اضافه کردن مدل جدید
در `src/config.py` به `MODELS_CONFIG` اضافه کنید:
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

### استفاده از دیتای شخصی
فایل CSV خود را در `data/raw/` قرار دهید و در `config.py`:
```python
TARGET_COLUMN = "your_fraud_column"
DROP_COLUMNS = ["columns_to_drop"]
```

---

## 📝 نکات مهم

1. **نسبت کلاس**: دیتای اصلی بسیار ناهمسان است (Fraud: 0.17%). SMOTE در پیش‌پردازش فعال است.
2. **ویژگی‌ها**: ویژگی‌های V1-V28 نتیجه PCA هستند (دیتا ناشناس). تنها `Amount` و `Time` اصلی هستند.
3. **Time**: معمولاً برای تشخیص تقلب مفید نیست و به طور پیش‌فرض حذف می‌شود.
4. **آستانه**: 0.5 پیش‌فرض است. برای تولید از `plot_threshold_analysis` برای یافتن بهینه استفاده کنید.
5. **استنتاج سریع**: مدل و اسکیلر در `models/` ذخیره شده‌اند و بدون نیاز به بازآموزی بارگذاری می‌شوند.

---

## 📄 لایسنس

MIT License - آزاد برای استفاده، تغییر و توزیع.

---

## 🤝 مشارکت

Pull Request ها خوش‌آمدند! لطفاً:
1. Fork کنید
2. Branch جدید بسازید (`git checkout -b feature/amazing-feature`)
3. تغییرات را Commit کنید (`git commit -m 'Add amazing feature'`)
4. Push کنید (`git push origin feature/amazing-feature`)
5. Pull Request باز کنید

---

**توسعه‌دهنده:** Bank Fraud Detection Team  
**ورژن:** 1.0.0