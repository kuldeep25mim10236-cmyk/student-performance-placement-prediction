"""
visualize.py
--------------
Reporting/analytics module: generates the analytics charts used inside the
Streamlit dashboard and the project report (CGPA distribution, placement
rate by branch, feature importance, etc.).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_cgpa_distribution(df: pd.DataFrame, out_path: Path = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["cgpa"], bins=20, color="#4C72B0", edgecolor="white")
    ax.set_title("CGPA Distribution")
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Number of Students")
    fig.tight_layout()
    out_path = out_path or OUT_DIR / "cgpa_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_placement_by_branch(df: pd.DataFrame, out_path: Path = None):
    rate = df.groupby("branch")["placed"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Placement Rate by Branch")
    ax.set_ylabel("Placement Rate (%)")
    ax.set_xlabel("Branch")
    fig.tight_layout()
    out_path = out_path or OUT_DIR / "placement_by_branch.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_importance(model, feature_names, title, out_path: Path):
    importances = model.feature_importances_
    order = importances.argsort()[::-1][:10]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#C44E52")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = Path(__file__).resolve().parent.parent / "data" / "students_dataset.csv"
    df = pd.read_csv(csv_path)
    plot_cgpa_distribution(df)
    plot_placement_by_branch(df)
    print("Saved analytics charts to", OUT_DIR)
