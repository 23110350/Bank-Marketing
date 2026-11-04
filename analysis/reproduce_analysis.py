"""Reproducible, leakage-aware Bank Marketing model comparison.

Run from the Bank-Marketing-ML-Project directory:
    python analysis/reproduce_analysis.py

The script deliberately excludes `duration`: it is only known after a call
and is unavailable for a pre-contact targeting decision. All learned
preprocessing is fitted inside each training fold through sklearn Pipelines.
"""
from pathlib import Path
import json
import warnings
import shutil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bank-full.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42


def make_preprocessor(numeric_columns, categorical_columns, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_columns),
        ("categorical", categorical_pipe, categorical_columns),
    ])


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA)
    required = {"y", "duration", "pdays"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    # Preserve the UCI observations. Do not delete zero-duration rows or
    # collapse pdays=-1 (never previously contacted) into pdays=0.
    df["previously_contacted"] = (df["pdays"] != -1).astype(int)
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y", "duration"])
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Models are specified before examining the held-out test set.
    model_specs = {
        "Majority baseline": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": (LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ), True),
        "Decision Tree": (DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=10,
            random_state=RANDOM_STATE
        ), False),
        "Random Forest": (RandomForestClassifier(
            n_estimators=40, max_depth=12, min_samples_leaf=10, max_samples=0.6, class_weight="balanced_subsample",
            n_jobs=1, random_state=RANDOM_STATE
        ), False),
        "XGBoost": (XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, eval_metric="logloss", n_jobs=1,
            random_state=RANDOM_STATE
        ), False),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows, prediction_data = [], {}
    for name, (estimator, scale) in model_specs.items():
        pipe = Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical, scale)),
            ("model", estimator),
        ])
        # Cross-validation uses training data only. AP is useful for the
        # imbalanced positive class; ROC-AUC is included as a secondary score.
        if name == "Majority baseline":
            cv_summary = {"cv_ap_mean": float(np.mean(cross_validate(
                pipe, X_train, y_train, cv=cv, scoring="average_precision",
                n_jobs=1, return_train_score=False)["test_score"]))}
        else:
            scores = cross_validate(
                pipe, X_train, y_train, cv=cv,
                scoring={"ap": "average_precision", "roc_auc": "roc_auc"},
                n_jobs=1, return_train_score=False,
            )
            cv_summary = {
                "cv_ap_mean": float(np.mean(scores["test_ap"])),
                "cv_ap_std": float(np.std(scores["test_ap"], ddof=1)),
                "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            }
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(X_test)[:, 1]
        else:
            prob = pred.astype(float)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        row = {
            "model": name,
            **cv_summary,
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "test_precision": float(precision_score(y_test, pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, prob)),
            "test_average_precision": float(average_precision_score(y_test, prob)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        rows.append(row)
        prediction_data[name] = {
            "y_true": y_test.astype(int).tolist(),
            "probability": np.asarray(prob, dtype=float).tolist(),
            "prediction_at_0_5": np.asarray(pred, dtype=int).tolist(),
        }
        print(f"{name:22s} accuracy={row['test_accuracy']:.4f} "
              f"F1={row['test_f1']:.4f} AP={row['test_average_precision']:.4f} "
              f"ROC-AUC={row['test_roc_auc']:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "model_comparison.csv", index=False)
    with open(OUT / "test_predictions.json", "w") as f:
        json.dump(prediction_data, f)
    website_data = ROOT / "website" / "data"
    website_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "test_predictions.json", website_data / "test_predictions.json")
    metadata = {
        "dataset": "UCI Bank Marketing bank-full.csv",
        "observations": int(len(df)),
        "positive_count": int(y.sum()),
        "negative_count": int((1-y).sum()),
        "positive_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "split": "stratified random 80/20 split",
        "cv": "3-fold stratified CV on training partition only",
        "target": "y (yes=1, no=0)",
        "excluded_feature": "duration (not available before the call)",
        "pdays_handling": "-1 retained; previously_contacted indicator added",
        "preprocessing": "imputation and one-hot encoding fitted inside each pipeline/fold",
        "resampling": "none; class weighting used for classifiers where supported",
        "note": "Test set used only for final reporting; not used for model selection.",
    }
    with open(OUT / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved outputs to: {OUT}")


if __name__ == "__main__":
    main()
