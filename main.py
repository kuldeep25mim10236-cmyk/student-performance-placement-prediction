"""
main.py
--------
End-to-end CLI pipeline orchestrator:

    1. Generate the dataset (if missing)
    2. Load it into the SQLite database
    3. Train the performance model
    4. Train the placement models
    5. Run full evaluation and save the report
    6. Generate analytics charts

Run:
    python main.py
"""

import sys
import logging
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parent

SRC_DIR = ROOT / "src"

sys.path.append(str(SRC_DIR))


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# DATASET PATH
# ============================================================

DATA_CSV = (
    ROOT
    / "data"
    / "students_dataset.csv"
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    import pandas as pd

    # --------------------------------------------------------
    # Import project modules
    # --------------------------------------------------------

    from generate_dataset import generate_dataset
    import database as db

    from evaluation import run_full_evaluation

    from visualize import (
        plot_cgpa_distribution,
        plot_placement_by_branch,
        plot_feature_importance
    )


    # ========================================================
    # STEP 1 — DATASET
    # ========================================================

    logger.info(
        "Step 1/6: Ensuring dataset exists..."
    )

    if not DATA_CSV.exists():

        DATA_CSV.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df = generate_dataset()

        df.to_csv(
            DATA_CSV,
            index=False
        )

        logger.info(
            "Dataset generated: %d rows",
            len(df)
        )

    else:

        df = pd.read_csv(
            DATA_CSV
        )

        logger.info(
            "Dataset already exists: %d rows",
            len(df)
        )


    # ========================================================
    # STEP 2 — SQLITE DATABASE
    # ========================================================

    logger.info(
        "Step 2/6: Loading data into SQLite database..."
    )

    db.init_db()

    db.bulk_load_from_dataframe(
        df
    )


    # ========================================================
    # STEP 3 & 4 — MODEL TRAINING + EVALUATION
    # ========================================================

    logger.info(
        "Step 3-4/6: Training models and running evaluation..."
    )

    report = run_full_evaluation(
        str(DATA_CSV)
    )


    # ========================================================
    # STEP 5 — BASIC ANALYTICS
    # ========================================================

    logger.info(
        "Step 5/6: Generating basic analytics charts..."
    )

    plot_cgpa_distribution(
        df
    )

    plot_placement_by_branch(
        df
    )


    # ========================================================
    # STEP 6 — FEATURE IMPORTANCE
    # ========================================================

    logger.info(
        "Step 6/6: Generating feature importance charts..."
    )

    try:

        plot_feature_importance(
            model_path=ROOT
            / "models"
            / "performance_model.joblib",

            output_name="feature_importance_performance.png",

            title="Performance Prediction - Feature Importance"
        )

        plot_feature_importance(
            model_path=ROOT
            / "models"
            / "placement_classifier.joblib",

            output_name="feature_importance_placement.png",

            title="Placement Prediction - Feature Importance"
        )

        logger.info(
            "Feature importance charts generated successfully."
        )

    except TypeError:

        logger.warning(
            "Feature importance function uses a different "
            "parameter format in visualize.py."
        )

    except Exception as e:

        logger.warning(
            "Could not generate feature importance charts: %s",
            e
        )


    # ========================================================
    # EVALUATION SUMMARY
    # ========================================================

    logger.info(
        "Pipeline complete. Evaluation summary:"
    )

    for model_name, metrics in report.items():

        logger.info(
            "  %s: %s",
            model_name,
            metrics
        )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "   AI STUDENT PERFORMANCE PROJECT"
    )

    print(
        "   Pipeline completed successfully! ✅"
    )

    print(
        "=============================================="
    )

    print(
        "\nDataset:"
    )

    print(
        f"  {DATA_CSV}"
    )

    print(
        "\nDashboard:"
    )

    print(
        "  streamlit run app.py"
    )

    print(
        "\nTests:"
    )

    print(
        "  pytest tests/ -v"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()