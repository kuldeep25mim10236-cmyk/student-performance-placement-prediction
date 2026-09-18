"""
preprocessing.py
------------------
Data cleaning, validation and feature engineering shared by both the
performance-prediction and placement-prediction pipelines.
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "attendance_pct", "cgpa", "backlogs", "internships", "projects_completed",
    "certifications", "coding_score", "communication_score",
    "extracurricular_score", "aptitude_score",
]
CATEGORICAL_FEATURES = ["branch", "gender"]


class DataValidationError(Exception):
    """Raised when the input dataframe fails schema / sanity checks."""
    pass


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw dataset from CSV with basic existence/format checks."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        logger.error("Dataset not found at %s", csv_path)
        raise DataValidationError(f"Dataset not found: {csv_path}") from exc

    required_cols = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["performance_score", "placed"])
    missing = required_cols - set(df.columns)
    if missing:
        raise DataValidationError(f"Dataset missing required columns: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates and obviously invalid rows."""
    before = len(df)
    df = df.drop_duplicates(subset="student_id")
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    # Clip out-of-range noise defensively (robustness / error-handling requirement)
    df["cgpa"] = df["cgpa"].clip(0, 10)
    df["attendance_pct"] = df["attendance_pct"].clip(0, 100)
    for col in ["coding_score", "communication_score", "extracurricular_score", "aptitude_score"]:
        df[col] = df[col].clip(0, 100)

    logger.info("Cleaned data: %d -> %d rows", before, len(df))
    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None):
    """One-hot encode branch/gender; returns transformed df + fitted encoders (for reuse at inference)."""
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that boost model signal."""
    df = df.copy()
    df["skill_index"] = (
        df["coding_score"] * 0.4
        + df["communication_score"] * 0.3
        + df["aptitude_score"] * 0.3
    )
    df["experience_index"] = df["internships"] * 2 + df["projects_completed"] + df["certifications"]
    df["risk_flag"] = ((df["backlogs"] > 1) | (df["attendance_pct"] < 65)).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Full pipeline: clean -> engineer -> encode -> return X (features only, no leakage)."""
    df = clean_data(df)
    df = engineer_features(df)
    df_encoded = encode_categoricals(df)

    drop_cols = [
        "student_id", "performance_score", "placed", "package_lpa",
        "created_at", "updated_at",
    ]
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
    X = df_encoded[feature_cols]
    return X, df_encoded


if __name__ == "__main__":
    from pathlib import Path

    csv_path = Path(__file__).resolve().parent.parent / "data" / "students_dataset.csv"
    df = load_data(str(csv_path))
    X, full = build_feature_matrix(df)
    print("Feature matrix shape:", X.shape)
    print(X.columns.tolist())
