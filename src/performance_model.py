"""
performance_model.py
----------------------
Module 2: Academic Performance Prediction
Trains a regression model that predicts a student's final performance score
(0-100) from academic and activity features.
"""

import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import load_data, build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "performance_model.joblib"


def train_performance_model(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Train and persist the RandomForestRegressor for performance-score prediction."""
    df = load_data(csv_path)
    X, full = build_feature_matrix(df)
    y = full["performance_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 3),
        "RMSE": round(mean_squared_error(y_test, y_pred) ** 0.5, 3),
        "R2": round(r2_score(y_test, y_pred), 3),
    }
    logger.info("Performance model metrics: %s", metrics)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, MODEL_PATH)
    logger.info("Saved performance model -> %s", MODEL_PATH)

    return model, metrics, list(X.columns)


def load_performance_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Performance model not trained yet. Run train_performance_model() first.")
    return joblib.load(MODEL_PATH)


def predict_performance(student_features: dict) -> float:
    """Predict performance score for a single new student (dict of raw feature values)."""
    bundle = load_performance_model()
    model, feature_columns = bundle["model"], bundle["feature_columns"]

    row = pd.DataFrame([student_features])
    row = pd.get_dummies(row, columns=["branch", "gender"])
    for col in feature_columns:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_columns]
    pred = model.predict(row)[0]
    return round(float(pred), 2)


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "students_dataset.csv"
    model, metrics, cols = train_performance_model(str(csv_path))
    print("Metrics:", metrics)
