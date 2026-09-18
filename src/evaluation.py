"""
evaluation.py
---------------
Consolidates evaluation of both the performance-regression model and the
placement-classification/regression models, and writes a JSON evaluation
report used in the project report and for monitoring model quality.
"""

import json
import logging
from pathlib import Path

from performance_model import train_performance_model
from placement_model import train_placement_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "evaluation_report.json"


def run_full_evaluation(csv_path: str) -> dict:
    """Train all models and collect their evaluation metrics into a single report."""
    logger.info("Training performance model...")
    _, perf_metrics, _ = train_performance_model(csv_path)

    logger.info("Training placement models...")
    _, _, clf_metrics, reg_metrics = train_placement_models(csv_path)

    report = {
        "performance_model": perf_metrics,
        "placement_classifier": clf_metrics,
        "package_regressor": reg_metrics,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report written -> %s", REPORT_PATH)
    return report


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "students_dataset.csv"
    report = run_full_evaluation(str(csv_path))
    print(json.dumps(report, indent=2))
