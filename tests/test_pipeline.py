"""
test_pipeline.py
------------------
Unit and validation tests for the data pipeline, database CRUD layer,
and trained models. Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import pandas as pd

from generate_dataset import generate_dataset
from preprocessing import clean_data, build_feature_matrix, DataValidationError
import database as db


@pytest.fixture(scope="module")
def sample_df():
    return generate_dataset(n=100, seed=1)


def test_generate_dataset_shape(sample_df):
    assert len(sample_df) == 100
    assert "performance_score" in sample_df.columns
    assert "placed" in sample_df.columns


def test_generate_dataset_value_ranges(sample_df):
    assert sample_df["cgpa"].between(0, 10).all()
    assert sample_df["attendance_pct"].between(0, 100).all()
    assert sample_df["placed"].isin([0, 1]).all()


def test_clean_data_removes_duplicates(sample_df):
    dup_df = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
    cleaned = clean_data(dup_df)
    assert len(cleaned) == len(sample_df)


def test_build_feature_matrix_no_leakage(sample_df):
    X, full = build_feature_matrix(sample_df)
    leak_cols = {"performance_score", "placed", "package_lpa", "student_id"}
    assert leak_cols.isdisjoint(set(X.columns))
    assert len(X) == len(sample_df)


def test_database_crud(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)

    record = {
        "student_id": "TEST001", "branch": "CSE", "gender": "Male",
        "attendance_pct": 85.0, "cgpa": 8.2, "backlogs": 0,
        "internships": 2, "projects_completed": 3, "certifications": 2,
        "coding_score": 75.0, "communication_score": 80.0,
        "extracurricular_score": 60.0, "aptitude_score": 70.0,
    }
    db.add_student(record, db_path)

    fetched = db.get_student("TEST001", db_path)
    assert fetched is not None
    assert fetched["cgpa"] == 8.2

    db.update_student("TEST001", {"cgpa": 9.0}, db_path)
    updated = db.get_student("TEST001", db_path)
    assert updated["cgpa"] == 9.0

    db.delete_student("TEST001", db_path)
    assert db.get_student("TEST001", db_path) is None


def test_database_validation_error(tmp_path):
    db_path = tmp_path / "test2.db"
    db.init_db(db_path)
    bad_record = {"student_id": "BAD001", "branch": "CSE"}  # missing required fields
    with pytest.raises(db.ValidationError):
        db.add_student(bad_record, db_path)


def test_load_data_missing_file():
    from preprocessing import load_data
    with pytest.raises(DataValidationError):
        load_data("nonexistent_file.csv")
