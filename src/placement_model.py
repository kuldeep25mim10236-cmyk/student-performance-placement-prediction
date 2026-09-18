"""
placement_model.py
---------------------
Module 3: Placement Prediction
Trains:
  (a) a classifier predicting placement likelihood (Placed / Not Placed)
  (b) a regressor predicting the expected package (LPA), conditioned on
      students who were historically placed.
"""

import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error,
)

from preprocessing import load_data, build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CLF_PATH = MODEL_DIR / "placement_classifier.joblib"
REG_PATH = MODEL_DIR / "package_regressor.joblib"


def train_placement_models(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    df = load_data(csv_path)
    X, full = build_feature_matrix(df)
    y_clf = full["placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=random_state, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    clf_metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1": round(f1_score(y_test, y_pred), 3),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    logger.info("Placement classifier metrics: %s", clf_metrics)

    # Package regressor: trained only on placed students
    placed_mask = full["placed"] == 1
    X_placed = X[placed_mask]
    y_placed = full.loc[placed_mask, "package_lpa"]

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X_placed, y_placed, test_size=test_size, random_state=random_state
    )
    reg = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=random_state, n_jobs=-1
    )
    reg.fit(Xp_train, yp_train)
    yp_pred = reg.predict(Xp_test)
    reg_metrics = {"MAE_LPA": round(mean_absolute_error(yp_test, yp_pred), 3)}
    logger.info("Package regressor metrics: %s", reg_metrics)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_columns": list(X.columns)}, CLF_PATH)
    joblib.dump({"model": reg, "feature_columns": list(X.columns)}, REG_PATH)
    logger.info("Saved placement classifier -> %s", CLF_PATH)
    logger.info("Saved package regressor -> %s", REG_PATH)

    return clf, reg, clf_metrics, reg_metrics


def _prepare_row(student_features: dict, feature_columns: list) -> pd.DataFrame:
    row = pd.DataFrame([student_features])
    row = pd.get_dummies(row, columns=["branch", "gender"])
    for col in feature_columns:
        if col not in row.columns:
            row[col] = 0
    return row[feature_columns]


def predict_placement(student_features: dict) -> dict:
    """Predict placement probability and, if likely placed, an expected package."""
    if not CLF_PATH.exists() or not REG_PATH.exists():
        raise FileNotFoundError("Placement models not trained yet. Run train_placement_models() first.")

    clf_bundle = joblib.load(CLF_PATH)
    reg_bundle = joblib.load(REG_PATH)

    clf, clf_cols = clf_bundle["model"], clf_bundle["feature_columns"]
    reg, reg_cols = reg_bundle["model"], reg_bundle["feature_columns"]

    row_clf = _prepare_row(student_features, clf_cols)
    proba = clf.predict_proba(row_clf)[0][1]
    label = "Placed" if proba >= 0.5 else "Not Placed"

    row_reg = _prepare_row(student_features, reg_cols)
    expected_package = round(float(reg.predict(row_reg)[0]), 2) if proba >= 0.5 else 0.0

    return {
        "placement_probability": round(float(proba), 3),
        "prediction": label,
        "expected_package_lpa": expected_package,
    }


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "students_dataset.csv"
    clf, reg, clf_metrics, reg_metrics = train_placement_models(str(csv_path))
    print("Classifier metrics:", clf_metrics)
    print("Regressor metrics:", reg_metrics)
