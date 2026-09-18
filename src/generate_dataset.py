"""
generate_dataset.py
--------------------
Generates a synthetic but statistically realistic dataset of student academic,
skill and activity records, along with derived labels for:
    1. Final Performance Score (regression target)
    2. Placement Status - Placed / Not Placed (classification target)
    3. Placement Package in LPA (regression target, only meaningful if placed)

The relationships between features and targets are seeded with realistic
correlations (e.g. higher CGPA + internships + projects raises placement
probability and package) plus random noise, so the downstream ML models have
genuine signal to learn from.

Run:
    python src/generate_dataset.py
Output:
    data/students_dataset.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_STUDENTS = 600

BRANCHES = ["CSE", "IT", "ECE", "MECH", "CIVIL", "EEE"]
GENDERS = ["Male", "Female"]


def generate_dataset(n=N_STUDENTS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    student_id = [f"STU{str(i).zfill(4)}" for i in range(1, n + 1)]
    branch = rng.choice(BRANCHES, size=n)
    gender = rng.choice(GENDERS, size=n)

    attendance = np.clip(rng.normal(80, 10, n), 45, 100).round(1)
    cgpa = np.clip(rng.normal(7.2, 1.1, n), 4.0, 10.0).round(2)
    backlogs = rng.poisson(0.6, n)
    backlogs = np.clip(backlogs, 0, 6)

    internships = rng.poisson(1.0, n)
    internships = np.clip(internships, 0, 5)

    projects = rng.poisson(2.0, n)
    projects = np.clip(projects, 0, 8)

    certifications = rng.poisson(1.5, n)
    certifications = np.clip(certifications, 0, 10)

    coding_score = np.clip(rng.normal(65, 15, n), 0, 100).round(1)          # e.g. HackerRank/LeetCode style score
    communication_score = np.clip(rng.normal(70, 12, n), 0, 100).round(1)   # soft-skill / mock-interview score
    extracurricular_score = np.clip(rng.normal(55, 20, n), 0, 100).round(1)
    aptitude_score = np.clip(rng.normal(68, 14, n), 0, 100).round(1)

    # ---- Derived target 1: Final Performance Score (0-100) ----
    performance_score = (
        cgpa * 6.0
        + attendance * 0.15
        + projects * 1.8
        + certifications * 1.2
        + coding_score * 0.15
        + aptitude_score * 0.10
        - backlogs * 3.5
        + rng.normal(0, 4, n)
    )
    performance_score = np.clip(performance_score, 0, 100).round(2)

    # ---- Derived target 2 & 3: Placement status + package ----
    placement_logit = (
        -11.8
        + cgpa * 0.9
        + internships * 0.55
        + projects * 0.30
        + certifications * 0.20
        + coding_score * 0.03
        + communication_score * 0.02
        + aptitude_score * 0.02
        - backlogs * 0.9
        + rng.normal(0, 0.8, n)
    )
    placement_prob = 1 / (1 + np.exp(-placement_logit))
    placed = (rng.random(n) < placement_prob).astype(int)

    base_package = (
        3.0
        + cgpa * 0.55
        + internships * 0.45
        + projects * 0.20
        + certifications * 0.15
        + coding_score * 0.03
        - backlogs * 0.35
        + rng.normal(0, 0.8, n)
    )
    package_lpa = np.where(placed == 1, np.clip(base_package, 3.0, 45.0), 0.0).round(2)

    df = pd.DataFrame(
        {
            "student_id": student_id,
            "branch": branch,
            "gender": gender,
            "attendance_pct": attendance,
            "cgpa": cgpa,
            "backlogs": backlogs,
            "internships": internships,
            "projects_completed": projects,
            "certifications": certifications,
            "coding_score": coding_score,
            "communication_score": communication_score,
            "extracurricular_score": extracurricular_score,
            "aptitude_score": aptitude_score,
            "performance_score": performance_score,
            "placed": placed,
            "package_lpa": package_lpa,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = Path(__file__).resolve().parent.parent / "data" / "students_dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} student records -> {out_path}")
    print(df.head())
    print("\nPlacement rate: {:.1f}%".format(df["placed"].mean() * 100))
