# 🏦 Loan Approval Predictor

A professional Streamlit dashboard for loan approval prediction using Logistic Regression and Decision Tree classifiers with SMOTE balancing and GridSearchCV tuning.

## 📁 Project Structure

```
loan_approval_app/
├── app.py                      ← Streamlit frontend
├── train.py                    ← Training + model saving script
├── requirements.txt
├── loan_approval_dataset.csv   ← Your dataset (place here)
└── models/                     ← Auto-created after training
    ├── best_model.joblib
    └── model_report.joblib
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train & save the model
```bash
python train.py
# or with a custom dataset path:
python train.py --data path/to/your_dataset.csv
```

This will:
- Load and preprocess the dataset
- Train Logistic Regression + Decision Tree (with SMOTE + GridSearchCV)
- Evaluate both models and pick the best
- Run 5-fold cross-validation
- Save `models/best_model.joblib` and `models/model_report.joblib`

### 3. Launch the app
```bash
streamlit run app.py
```

## 🗂 App Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Dataset KPIs and a sample preview |
| 📊 Data Explorer | Interactive charts: distributions, CIBIL analysis, correlations |
| 🔮 Predict | Live prediction form with confidence breakdown |
| 🤖 Model Info | Classification reports, confusion matrices, feature importance, learning curve |

## 📌 Notes

- The app expects the dataset columns to include: `loan_id`, `loan_status`, `no_of_dependents`, `education`, `self_employed`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`.
- Column names are automatically stripped and lowercased.
- Target labels are expected to be `Approved` / `Rejected` (case-insensitive).