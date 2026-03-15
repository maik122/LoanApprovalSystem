# LoanIQ — Loan Approval Prediction System

A machine learning system that predicts whether a loan application should be approved or rejected, packaged as a production-ready Streamlit dashboard. Built with scikit-learn, imbalanced-learn, and Plotly.

---

## What It Does

LoanIQ takes an applicant's financial profile — CIBIL score, income, loan amount, assets, employment status — and predicts approval outcome using a trained Decision Tree or Logistic Regression model. Both models are tuned with GridSearchCV and trained on SMOTE-balanced data to handle class imbalance.

The app has four pages:

| Page | What you get |
|------|-------------|
| **Overview** | Dataset KPIs, approval split donut chart, CIBIL score distribution |
| **Explorer** | Interactive violin plots, categorical breakdowns, CIBIL band approval rates, scatter plots, correlation matrix |
| **Predict** | Live prediction form with confidence gauge and probability breakdown |
| **Model** | Confusion matrices, classification reports, feature importance chart, learning curve |

---

## Who It's For

- **Loan officers** who need a fast pre-screening tool before manual review
- **Credit risk analysts** who want to understand what drives approvals in their portfolio
- **Fintech teams** building an in-house scoring baseline before investing in enterprise tools
- **Data science students** who want a complete, working reference project covering EDA → training → deployment

---

## Project Structure

```
LoanApprovalSystem/
│
├── app.py                       ← Streamlit dashboard (run this)
├── train.py                     ← Training script (run this first)
├── requirements.txt             ← Python dependencies
├── README.md                    ← You are here
│
├── loan_approval_dataset.csv    ← Your dataset (place here)
│
└── models/                      ← Auto-created after running train.py
    ├── best_model.joblib        ← Saved best pipeline (SMOTE + classifier)
    ├── encoder.joblib           ← Saved OneHotEncoder for categoricals
    └── model_report.joblib      ← Evaluation metrics, confusion matrices, feature importances
```

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your dataset

Put `loan_approval_dataset.csv` in the same folder as `app.py`. The file should contain these columns (column names are auto-cleaned — spaces and casing don't matter):

| Column | Type | Description |
|--------|------|-------------|
| `loan_id` | string | Unique identifier (dropped during training) |
| `no_of_dependents` | int | Number of financial dependents |
| `education` | string | `Graduate` or `Not Graduate` |
| `self_employed` | string | `Yes` or `No` |
| `income_annum` | int | Annual income in ₹ |
| `loan_amount` | int | Requested loan amount in ₹ |
| `loan_term` | int | Loan term in months |
| `cibil_score` | int | Credit score (300–900) |
| `residential_assets_value` | int | Value of residential assets in ₹ |
| `commercial_assets_value` | int | Value of commercial assets in ₹ |
| `luxury_assets_value` | int | Value of luxury assets in ₹ |
| `bank_asset_value` | int | Value of bank assets in ₹ |
| `loan_status` | string | `Approved` or `Rejected` (target variable) |

### 3. Train the model

```bash
python train.py
```

To use a custom dataset path:

```bash
python train.py --data path/to/your_dataset.csv
```

This will:
- Encode categorical features with `OneHotEncoder`
- Split data 80/20 with stratification
- Train both Logistic Regression and Decision Tree with SMOTE oversampling
- Tune both models with 5-fold GridSearchCV
- Pick the best model by macro F1
- Run cross-validation on the full dataset
- Save three files into `models/`

Expected output:
```
┌──────────────────────────┐
│  1 · Loading Dataset     │
└──────────────────────────┘
   Shape : (4269, 13)
   Target: {'Approved': 2656, 'Rejected': 1613}
...
   ✅ Model   → models/best_model.joblib
   ✅ Encoder → models/encoder.joblib
   ✅ Report  → models/model_report.joblib
```

### 4. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The top-right corner shows a green **● Live** badge when the model is loaded correctly.

---

## How the ML Pipeline Works

```
Raw CSV
   │
   ▼
OneHotEncoder          ← categorical: education, self_employed
   +                   ← numeric: passed through as-is
NumPy array
   │
   ▼
Train / Test Split (80/20, stratified)
   │
   ├──▶ SMOTE (training set only)  ← handles class imbalance
   │         │
   │         ▼
   │    Classifier
   │    ├── Logistic Regression  (GridSearchCV over C, solver)
   │    └── Decision Tree        (GridSearchCV over depth, alpha, leaf size)
   │
   ▼
Best model selected by macro F1
   │
   ▼
5-fold cross-validation on full dataset
   │
   ▼
Save model + encoder + report
```

The encoder is saved **separately** from the pipeline (not inside a `ColumnTransformer`) to avoid scikit-learn version compatibility issues when loading across environments.

---

## Model Performance

Typical results on the default dataset:

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression (tuned) | ~79% | ~0.78 |
| Decision Tree (tuned) | ~98% | ~0.98 |

**Key finding:** CIBIL score is by far the strongest predictor, followed by loan amount, income, and loan term. Asset values and employment status have marginal impact.

---

## Retraining on New Data

Just delete the `models/` folder and rerun `train.py`:

```bash
rm -rf models/
python train.py --data new_dataset.csv
```

Then restart the Streamlit app (Ctrl+C → `streamlit run app.py`). The `@st.cache_resource` decorator caches the model in memory, so a browser refresh alone is not enough after retraining.

---

## Limitations & Important Caveats

- **Not a substitute for human judgment.** This tool is intended as a decision-support aid. Final lending decisions should always involve human review.
- **Regulatory compliance.** Automated credit decisions are regulated in many jurisdictions (EU GDPR Article 22, US ECOA/FCRA). Before using this in any real lending context, consult legal and compliance teams.
- **Model fairness.** The model has not been audited for demographic bias. Features like income and employment can be proxies for protected characteristics.
- **Not production-scale.** Streamlit is suitable for internal tools and demos. High-volume real-time scoring would require a proper API backend (FastAPI, Flask) with a model serving layer.
- **Dataset dependency.** Performance degrades if the distribution of new applications differs significantly from the training data. Retrain periodically as your portfolio grows.

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
```

---

## License

