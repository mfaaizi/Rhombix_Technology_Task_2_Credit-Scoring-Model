#!/usr/bin/env python3
"""
Credit Scoring Model Training Script (Professional)
Author: Muhammad Faaiz
Date: 2025-08-18
"""

import argparse
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


# -------------------------
# Logging
# -------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("credit-scoring")


# -------------------------
# Data Loading & Prep
# -------------------------
def load_data(loan_train: str) -> pd.DataFrame:
    """Load CSV and perform lightweight cleaning + feature engineering."""
    df = pd.read_csv(loan_train)
    logger = setup_logging()
    logger.info(f"Loaded data: {loan_train} (rows={len(df)}, cols={len(df.columns)})")

    # Strip column names
    df.columns = df.columns.str.strip()

    # Harmonize 'Dependents' (e.g., '3+' -> 3)
    if "Dependents" in df.columns:
        df["Dependents"] = (
            df["Dependents"]
            .astype(str)
            .str.replace("+", "", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )

    # Basic imputations for obvious numeric holes (safe defaults)
    for col in ["Credit_History", "Loan_Amount_Term"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["DTI"] = df["LoanAmount"] / (df["Total_Income"] + 1e-6)  # debt-to-income

    # Target: map Loan_Status (Y/N) -> creditworthy flag (1/0)
    if "Loan_Status" not in df.columns:
        raise ValueError("Expected column 'Loan_Status' (Y/N) not found.")
    df["creditworthy"] = df["Loan_Status"].map({"Y": 1, "N": 0}).astype(int)

    # Drop obvious IDs
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    return df


# -------------------------
# Modeling Pipeline
# -------------------------
def build_pipeline(numeric_cols, categorical_cols, scale_pos_weight=1.0) -> Pipeline:
    """Create preprocessing + XGBoost pipeline."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    return pipe


def tune_hyperparams(pipe: Pipeline, X_train, y_train) -> Pipeline:
    """Grid search over a compact, high-signal hyperparam grid."""
    param_grid = {
        "classifier__n_estimators": [300, 500],
        "classifier__learning_rate": [0.03, 0.07],
        "classifier__max_depth": [3, 4, 5],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
        "classifier__reg_lambda": [0.5, 1.0, 2.0],
    }

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    logger = setup_logging()
    logger.info("Starting hyperparameter tuning...")
    gs.fit(X_train, y_train)
    logger.info(f"Best ROC-AUC (CV): {gs.best_score_:.4f}")
    logger.info(f"Best Params: {gs.best_params_}")
    return gs.best_estimator_


# -------------------------
# Evaluation (with threshold tuning)
# -------------------------
def evaluate(model: Pipeline, X_val, y_val, label="Validation"):
    """Evaluate with default 0.5 threshold and tuned threshold (max F1)."""
    proba = model.predict_proba(X_val)[:, 1]

    # Global metrics
    roc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    
    # Default threshold
    pred_05 = (proba >= 0.5).astype(int)
    
    # Tune threshold for max F1 on this split
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    # Avoid division by zero
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    best_idx = int(np.argmax(f1))
    best_threshold = 0.5 if best_idx >= len(thresholds) else thresholds[best_idx]
    pred_best = (proba >= best_threshold).astype(int)
    
    # Calculate additional metrics
    accuracy_05 = accuracy_score(y_val, pred_05)
    f1_05 = f1_score(y_val, pred_05)
    accuracy_best = accuracy_score(y_val, pred_best)
    f1_best = f1_score(y_val, pred_best)
    
    # Confusion matrices
    cm_05 = confusion_matrix(y_val, pred_05)
    cm_best = confusion_matrix(y_val, pred_best)
    
    # Classification reports
    cr_05 = classification_report(y_val, pred_05, output_dict=True)
    cr_best = classification_report(y_val, pred_best, output_dict=True)
    
    return {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "best_threshold": best_threshold,
        "metrics_05": {
            "accuracy": accuracy_05,
            "f1": f1_05,
            "confusion_matrix": cm_05.tolist(),
            "classification_report": cr_05
        },
        "metrics_best": {
            "accuracy": accuracy_best,
            "f1": f1_best,
            "confusion_matrix": cm_best.tolist(),
            "classification_report": cr_best
        },
        "precision_recall_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist()
        }
    }


# -------------------------
# Utilities
# -------------------------
def extract_feature_importance(model: Pipeline, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe of feature importances aligned to transformed columns."""
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    # Get transformed feature names
    num_names = pre.transformers_[0][2]
    cat_names = pre.named_transformers_["cat"]["onehot"].get_feature_names_out(
        pre.transformers_[1][2]
    )
    feature_names = np.concatenate([num_names, cat_names])

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        # Fallback (rare for XGB); uniform zeros
        importances = np.zeros(len(feature_names))

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return fi


def predict_new_data(model, data, threshold=0.5):
    """Predict on new data with adjustable threshold."""
    if isinstance(data, pd.DataFrame):
        # If it's a DataFrame, assume it's already processed
        probabilities = model.predict_proba(data)[:, 1]
    else:
        # If it's not a DataFrame, assume it's the raw data that needs preprocessing
        probabilities = model.predict_proba(data)[:, 1]
    
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train a Credit Scoring Model")
    p.add_argument("--input", type=str, required=True, help="Path to input CSV")
    p.add_argument("--output", type=str, default="credit_model.pkl", help="Path to save trained model")
    p.add_argument("--featimp", type=str, default="feature_importances.csv", help="Path to save feature importances CSV")
    p.add_argument("--test_size", type=float, default=0.2, help="Test size ratio")
    return p.parse_args()


def train_model(input_path, test_size=0.2):
    """Train model function that can be called from GUI."""
    logger = setup_logging()
    
    # Load
    df = load_data(input_path)

    # Features / target
    y = df["creditworthy"].values
    drop_cols = ["Loan_Status", "creditworthy"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Class imbalance weight
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = float(neg) / float(pos) if pos > 0 else 1.0
    logger.info(f"Class balance (train): pos={pos}, neg={neg}, scale_pos_weight={spw:.3f}")

    # Column groups
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    # Build, tune, fit
    base_pipe = build_pipeline(numeric_cols, categorical_cols, scale_pos_weight=spw)
    model = tune_hyperparams(base_pipe, X_train, y_train)

    # Evaluate
    logger.info("Evaluating on validation (test) split...")
    evaluation_results = evaluate(model, X_test, y_test, label="Test")

    # Feature importances
    fi = extract_feature_importance(model, X_train)
    
    # Prepare model metadata
    model_data = {
        "pipeline": model,
        "best_threshold": evaluation_results["best_threshold"],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "evaluation_results": evaluation_results,
        "feature_importances": fi.to_dict(orient="records"),
        "model_info": {
            "author": "Muhammad Faaiz",
            "date": "2025-08-18",
            "algorithm": "XGBoost (GridSearchCV tuned)",
            "training_data_shape": X_train.shape,
            "test_data_shape": X_test.shape,
            "class_balance": {"positive": pos, "negative": neg, "scale_pos_weight": spw}
        }
    }
    
    return model_data, X_test, y_test, fi


def main():
    args = parse_args()
    logger = setup_logging()
    
    model_data, X_test, y_test, fi = train_model(args.input, args.test_size)
    
    # Save model and metadata
    joblib.dump(model_data, args.output)
    logger.info(f"Saved trained model + metadata to: {args.output}")

    # (Optional) also save feature importances as CSV
    fi.to_csv(args.featimp, index=False)
    logger.info(f"Saved feature importances to: {args.featimp}")


if __name__ == "__main__":
    main()