# 💳 Credit Card Fraud Detection

![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

An end-to-end machine learning system to detect fraudulent credit card transactions using both supervised classification and unsupervised anomaly detection — with an interactive Streamlit dashboard.

---

## 📌 Problem Statement

Credit card fraud is a critical real-world problem. The challenge is extreme class imbalance — only **0.17%** of transactions are fraudulent. Standard accuracy metrics are misleading; the focus must be on **Precision, Recall, F1-score, and AUC-PR**.

---

## 📁 Project Structure

```
fraud_detection/
│
├── fraud_dashboard.py          # Streamlit interactive dashboard
├── fraud_notebook.ipynb        # Step-by-step Jupyter notebook
├── fraud_model.pkl             # Saved XGBoost model (generated after training)
├── scaler.pkl                  # Saved StandardScaler (generated after training)
├── threshold.pkl               # Saved optimal threshold (generated after training)
├── requirements.txt            # All dependencies
├── .gitignore
└── README.md
```

> ⚠️ `creditcard.csv` is **not included** — download it separately (see below).

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Features | 31 (V1–V28 PCA + Amount + Time + Class) |

### Feature Description

| Feature | Description |
|---|---|
| `V1` – `V28` | PCA-transformed components (anonymised for privacy) |
| `Amount` | Transaction amount in USD |
| `Time` | Seconds elapsed since first transaction in the dataset |
| `Class` | Target — `0` = Normal, `1` = Fraud |

### How to Download the Dataset
1. Go to [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the project root folder

---

## 🧠 Models Used

| Model | Type | Description |
|---|---|---|
| Logistic Regression | Supervised | Baseline classifier with SMOTE |
| Balanced Random Forest | Supervised | Under-samples majority class per tree |
| XGBoost | Supervised | Best performer — uses `scale_pos_weight` |
| Isolation Forest | Unsupervised | Detects fraud as anomalies, no labels needed |

---

## ⚙️ Key Techniques

**Handling class imbalance:**
- `scale_pos_weight` in XGBoost — weights fraud class inversely to its frequency
- `BalancedRandomForestClassifier` — auto under-samples per tree, faster than SMOTE
- `SMOTE` — synthetic oversampling, used for Logistic Regression baseline
- `class_weight='balanced'` — adjusts sample weights automatically

**Evaluation strategy:**
- Precision-Recall AUC instead of ROC-AUC (more honest for imbalanced data)
- Decision threshold tuning — finds the threshold that maximises F1-score on the fraud class
- Confusion matrix analysis — focus on minimising False Negatives (missed frauds)

---

## 📈 Results

| Model | ROC-AUC | Avg Precision (AP) |
|---|---|---|
| Logistic Regression | ~0.97 | ~0.72 |
| Balanced Random Forest | ~0.98 | ~0.85 |
| XGBoost | ~0.98 | ~0.87 |
| Isolation Forest | ~0.92 | ~0.28 |

> Results may vary slightly based on random seed and dataset version.

---

## 🖥️ Streamlit Dashboard

The dashboard has **5 interactive tabs:**

| Tab | Contents |
|---|---|
| 📊 EDA | Class imbalance · Amount & time distributions · Feature correlations with fraud |
| 🏆 Model Performance | ROC curves · Confusion matrices · Classification reports · Threshold tuning chart |
| 📈 Precision-Recall | PR curves for all models · Precision vs Recall tradeoff · AP score table |
| 🔍 Feature Importance | Top-N XGBoost importances with adjustable slider · Fraud signal insights |
| 🎯 Predict Transaction | Enter V1–V28 + Amount + Time → fraud probability + BLOCK / ALLOW decision |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Place `creditcard.csv` in the project root (see Dataset section above).

### 4. Run the Streamlit dashboard
```bash
streamlit run fraud_dashboard.py
```

Opens at `http://localhost:8501` — upload `creditcard.csv` using the sidebar.
Models train automatically in ~60 seconds.

### 5. (Optional) Run the Jupyter notebook
```bash
jupyter notebook fraud_notebook.ipynb
```

Run cells top to bottom — each cell is self-contained with comments.

---

## 💾 Save Trained Models

Run this in the final notebook cell after training:

```python
import joblib, os, numpy as np
from sklearn.metrics import f1_score

os.makedirs('fraud_detection', exist_ok=True)

# Save model and scaler
joblib.dump(xgb_model, 'fraud_model.pkl')
joblib.dump(scaler,    'scaler.pkl')

# Save best threshold
thresholds = np.arange(0.1, 0.9, 0.01)
f1s = [f1_score(y_test, (y_prob_xgb >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1s)]
joblib.dump(best_thresh, 'threshold.pkl')

print(f"Saved — best threshold: {best_thresh:.2f}")
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
streamlit>=1.20.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧪 Concepts Covered

- Binary classification on severely imbalanced data
- Anomaly detection using Isolation Forest
- SMOTE and class weighting strategies
- Decision threshold tuning for maximum F1-score
- Precision-Recall curve analysis
- XGBoost with `scale_pos_weight`
- Feature importance on PCA-transformed features
- Streamlit dashboard with real-time prediction
- Model serialisation with `joblib`

---

## 🗂️ .gitignore

```
creditcard.csv
*.pkl
__pycache__/
.ipynb_checkpoints/
*.pyc
.env
```

---

## 👤 Author

**Rohan**
- GitHub: [Rohan-ai24](https://github.com/Rohan-ai24/Credit-Card-Fraud-Detection)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Dataset: [Andrea Dal Pozzolo et al. — ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [imbalanced-learn](https://imbalanced-learn.org/) — BalancedRandomForest & SMOTE
- [Streamlit](https://streamlit.io/) — dashboard framework
