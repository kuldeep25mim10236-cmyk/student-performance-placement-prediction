# 🎓 AI-Based Student Performance & Placement Prediction System

An end-to-end machine learning system that predicts a student's **academic
performance score** and **campus placement outcome** (placed/not placed +
expected package), backed by a SQLite data layer and an interactive
Streamlit dashboard.

Built for the **VITyarthi "Build Your Own Project"** flipped-course
evaluation.

---

## Overview

Colleges and training & placement cells struggle to identify, early enough,
which students are at risk of poor academic performance or of missing
placement opportunities. This project builds a data-driven system that:

1. Stores and manages student academic/activity records (CRUD)
2. Predicts a student's **final performance score** (0–100) using a
   regression model trained on academic and engagement features
3. Predicts **placement likelihood and expected package (LPA)** using a
   classification + regression model pipeline
4. Surfaces analytics (placement rate by branch, CGPA distribution, model
   feature importance) through an interactive dashboard

---

## Features

- **Student Data Management** — Add, view, update, delete student records
  via a validated SQLite-backed CRUD layer (`src/database.py`)
- **Performance Prediction** — RandomForestRegressor predicts a 0–100
  performance score from CGPA, attendance, backlogs, projects,
  certifications, coding/aptitude/communication scores, etc.
- **Placement Prediction** — RandomForestClassifier predicts placement
  probability; a second RandomForestRegressor estimates the expected
  package (LPA) for likely-placed students
- **Analytics Dashboard** — placement rate by branch, CGPA distribution,
  feature-importance charts, live dataset KPIs
- **Full evaluation pipeline** — MAE/RMSE/R² for regression,
  Accuracy/Precision/Recall/F1/Confusion-Matrix for classification, saved
  to `docs/evaluation_report.json`
- **Tested** — 7 unit/validation tests (`pytest tests/`) covering data
  generation, cleaning, feature engineering (no target leakage), and CRUD
  operations including validation errors

---

## Technologies / Tools Used

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn (RandomForestRegressor, RandomForestClassifier) |
| Data | pandas, numpy |
| Storage | SQLite (`sqlite3`) |
| Dashboard / UI | Streamlit |
| Visualization | matplotlib |
| Testing | pytest |
| Model persistence | joblib |
| Diagrams | Graphviz, matplotlib |
| Version control | Git |

---

## Project Structure

```
student-performance-placement-prediction/
├── app.py                       # Streamlit dashboard (main entry point for UI)
├── main.py                      # End-to-end CLI pipeline orchestrator
├── requirements.txt
├── README.md
├── statement.md
├── src/
│   ├── generate_dataset.py      # Synthetic dataset generator
│   ├── database.py              # Module 1: Student Data Management (CRUD)
│   ├── preprocessing.py         # Data cleaning + feature engineering
│   ├── performance_model.py     # Module 2: Performance Prediction
│   ├── placement_model.py       # Module 3: Placement Prediction
│   ├── evaluation.py            # Consolidated model evaluation
│   └── visualize.py             # Analytics chart generation
├── data/
│   └── students_dataset.csv     # Generated dataset (600 records)
├── models/                      # Saved model artifacts (*.joblib)
├── docs/
│   ├── diagrams/                # Architecture, UML, ER & analytics diagrams
│   └── evaluation_report.json   # Model evaluation metrics
├── tests/
│   └── test_pipeline.py         # Unit & validation tests
└── Project_Report.pdf           # Full submission report
```

---

## Steps to Install & Run

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd student-performance-placement-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the full pipeline (generates data, trains models, evaluates)

```bash
python main.py
```

This will:
- Generate `data/students_dataset.csv` (if not already present)
- Load records into `data/students.db`
- Train and save the performance and placement models to `models/`
- Run evaluation and write `docs/evaluation_report.json`
- Generate analytics charts into `docs/diagrams/`

### 3. Launch the interactive dashboard

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

---

## Instructions for Testing

Run the automated test suite:

```bash
pytest tests/ -v
```

Covers:
- Dataset generation shape & value-range validity
- Duplicate/missing-value cleaning
- Feature matrix construction (verifies no target leakage)
- Full student CRUD lifecycle (create/read/update/delete)
- CRUD input validation error handling
- Missing-file error handling in the data loader

To manually verify model quality, inspect `docs/evaluation_report.json`
after running `python main.py`, or re-run individual modules:

```bash
python src/performance_model.py
python src/placement_model.py
```

---

## Screenshots

See `docs/diagrams/` for the dashboard's analytics output (CGPA
distribution, placement rate by branch, feature importance) and the full
system diagrams (architecture, workflow, UML, ER).

---

## Model Performance (on held-out test split)

| Model | Metric | Value |
|---|---|---|
| Performance Regressor | R² | 0.65 |
| Performance Regressor | MAE | 4.37 |
| Placement Classifier | Accuracy | 0.68 |
| Placement Classifier | F1-score | 0.66 |
| Package Regressor | MAE (LPA) | 0.77 |

(Full metrics with confusion matrix in `docs/evaluation_report.json`.)
