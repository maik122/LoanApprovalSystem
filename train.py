"""
train.py — Train, tune, evaluate and SAVE the best loan approval model.

Usage:
    python train.py
    python train.py --data path/to/dataset.csv
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score,
    learning_curve, train_test_split,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR   = Path("models")
MODEL_PATH  = MODEL_DIR / "best_model.joblib"
OHE_PATH    = MODEL_DIR / "ohe.joblib"          # saved separately — no ColumnTransformer
REPORT_PATH = MODEL_DIR / "model_report.joblib"

TARGET      = "loan_status"
ID_COLS     = ["loan_id"]
CAT_COLS    = ["education", "self_employed"]
NUM_COLS    = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]


def banner(text):
    line = "─" * (len(text) + 4)
    print(f"\n┌{line}┐\n│  {text}  │\n└{line}┘")


def load_data(path):
    banner("1 · Loading Dataset")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    print(f"   Shape : {df.shape}")
    print(f"   Target: {df[TARGET].value_counts().to_dict()}")
    return df


def preprocess(df, ohe=None, fit=False):
    """
    Encode categoricals with OHE and concatenate with numerics.
    Returns (X_array, feature_names, fitted_ohe).
    No ColumnTransformer — avoids joblib version mismatch entirely.
    """
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    num_cols = [c for c in NUM_COLS  if c in df.columns]

    num_part = df[num_cols].values.astype(float)

    if fit:
        ohe = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
        cat_part = ohe.fit_transform(df[cat_cols])
    else:
        cat_part = ohe.transform(df[cat_cols])

    X = np.hstack([cat_part, num_part])

    cat_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = cat_names + num_cols

    return X, feature_names, ohe


def build_pipelines():
    log_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf",   LogisticRegression(max_iter=2000)),
    ])
    tree_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("clf",   DecisionTreeClassifier(random_state=42)),
    ])
    return log_pipe, tree_pipe


LOG_PARAM_GRID = {
    "clf__C":       [0.1, 0.5, 1.0, 2.0, 5.0],
    "clf__solver":  ["liblinear", "lbfgs"],
}
TREE_PARAM_GRID = {
    "clf__max_depth":         [None, 4, 6, 8, 12],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__min_samples_leaf":  [1, 2, 5, 10],
    "clf__ccp_alpha":         [0.0, 0.001, 0.01],
}


def train_and_tune(X_train, y_train):
    banner("3 · Hyperparameter Tuning (5-fold CV)")
    log_pipe, tree_pipe = build_pipelines()

    log_grid = GridSearchCV(log_pipe, LOG_PARAM_GRID,
                            scoring="f1_macro", cv=5, n_jobs=-1, verbose=0)
    log_grid.fit(X_train, y_train)
    print(f"   LogReg best params : {log_grid.best_params_}")
    print(f"   LogReg best CV F1  : {log_grid.best_score_:.4f}")

    tree_grid = GridSearchCV(tree_pipe, TREE_PARAM_GRID,
                             scoring="f1_macro", cv=5, n_jobs=-1, verbose=0)
    tree_grid.fit(X_train, y_train)
    print(f"   Tree   best params : {tree_grid.best_params_}")
    print(f"   Tree   best CV F1  : {tree_grid.best_score_:.4f}")

    return log_grid.best_estimator_, tree_grid.best_estimator_


def evaluate(models, X_test, y_test, labels):
    banner("4 · Evaluation on Hold-out Test Set")
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1  = f1_score(y_test, y_pred, average="macro")
        rep = classification_report(y_test, y_pred, digits=4)
        cm  = confusion_matrix(y_test, y_pred, labels=labels)
        results[name] = {"f1": f1, "report": rep, "cm": cm, "model": model}
        print(f"\n  [{name}]  F1 macro = {f1:.4f}\n{rep}")
    return results


def main(data_path):
    # 1. Load
    df = load_data(data_path)

    # 2. Preprocess (fit OHE on full df, then split)
    banner("2 · Preprocessing & Split")
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    num_cols = [c for c in NUM_COLS  if c in df.columns]

    X_df = df.drop(columns=ID_COLS + [TARGET], errors="ignore")
    y    = df[TARGET].astype(str).str.strip()
    labels = sorted(y.unique())

    # Fit OHE on full dataset before split (no label leakage — OHE is unsupervised)
    X_all, feature_names, fitted_ohe = preprocess(X_df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"   Train : {X_train.shape}   Test: {X_test.shape}")
    print(f"   Class balance (train): {y_train.value_counts().to_dict()}")

    # 3. Train & tune (pipelines are now just SMOTE + clf — no ColumnTransformer)
    log_best, tree_best = train_and_tune(X_train, y_train)

    # 4. Evaluate
    results = evaluate(
        {"Logistic Regression (Tuned)": log_best, "Decision Tree (Tuned)": tree_best},
        X_test, y_test, labels,
    )

    # 5. Pick best
    best_name  = max(results, key=lambda k: results[k]["f1"])
    best_f1    = results[best_name]["f1"]
    best_model = results[best_name]["model"]
    banner(f"Best Model → {best_name}  (F1={best_f1:.4f})")

    # 6. Cross-val on full preprocessed data
    banner("5 · Cross-validation (5-fold, full dataset)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_all, y, cv=cv,
                                scoring="f1_macro", n_jobs=-1)
    print(f"   CV F1 scores : {cv_scores.round(4)}")
    print(f"   Mean ± Std   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 7. Feature importances
    fi = None
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        fi = dict(zip(feature_names, clf.feature_importances_))
        top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]
        print("\n   Top-5 features:")
        for feat, imp in top5:
            print(f"     {feat:<40} {imp:.4f}")

    # 8. Learning curve
    banner("6 · Learning Curve")
    sizes, tr, te = learning_curve(
        best_model, X_all, y,
        cv=5, scoring="f1_macro", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 7),
    )
    lc = {
        "train_sizes": sizes.tolist(),
        "train_mean":  tr.mean(axis=1).tolist(),
        "train_std":   tr.std(axis=1).tolist(),
        "test_mean":   te.mean(axis=1).tolist(),
        "test_std":    te.std(axis=1).tolist(),
    }

    # 9. Save
    banner("7 · Saving Artifacts")
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)   # pipeline: SMOTE + clf only
    joblib.dump(fitted_ohe, OHE_PATH)     # OHE saved separately
    joblib.dump({
        "best_model_name":    best_name,
        "best_f1":            best_f1,
        "cv_f1_mean":         cv_scores.mean(),
        "cv_f1_std":          cv_scores.std(),
        "model_reports":      {k: v["report"] for k, v in results.items()},
        "confusion_matrices": {k: v["cm"].tolist() for k, v in results.items()},
        "labels":             labels,
        "feature_names":      feature_names,
        "feature_importances":fi,
        "learning_curve":     lc,
        "cat_cols":           cat_cols,
        "num_cols":           num_cols,
    }, REPORT_PATH)

    print(f"\n   ✅ Model  → {MODEL_PATH}")
    print(f"   ✅ OHE    → {OHE_PATH}")
    print(f"   ✅ Report → {REPORT_PATH}")
    banner("Done 🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="loan_approval_dataset.csv")
    args = parser.parse_args()
    main(args.data)