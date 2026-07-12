"""
eda.py
------
Exploratory Data Analysis for the QUASAR training dataset.

Generates descriptive statistics, correlation heatmaps, histograms,
boxplots, scatter plots, and an ML-readiness summary.

Run:
    python -m src.eda
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAS_SNS = True
except ImportError:
    HAS_SNS = False

DATASET = Path("datasets/training_dataset.csv")
REPORT_DIR = Path("reports")
PLOTS = Path("plots")


def write_report(text: str = "") -> None:
    """Append *text* to the EDA report file."""
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(str(text) + "\n")


def section(title: str) -> None:
    """Write a section header to the report."""
    write_report()
    write_report("=" * 80)
    write_report(title)
    write_report("=" * 80)


def run_eda() -> None:
    """Run the full exploratory data analysis pipeline."""

    for p in [
        REPORT_DIR,
        PLOTS / "histograms",
        PLOTS / "boxplots",
        PLOTS / "scatterplots",
        PLOTS / "correlations",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "eda_report.txt"

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    df = pd.read_csv(DATASET)
    report_path.write_text("")

    # Dataset Overview
    section("DATASET OVERVIEW")
    write_report(f"Rows: {len(df)}")
    write_report(f"Columns: {len(df.columns)}")
    num = df.select_dtypes(include=np.number)
    cat = df.select_dtypes(exclude=np.number)
    write_report(f"Numeric: {len(num.columns)}")
    write_report(f"Categorical: {len(cat.columns)}")

    # Descriptive Statistics
    section("DESCRIPTIVE STATISTICS")
    stats = num.describe().T
    stats["variance"] = num.var()
    stats["skewness"] = num.skew()
    stats["kurtosis"] = num.kurtosis()
    write_report(stats.to_string())

    # Class Distribution
    section("CLASS DISTRIBUTION")
    if "reliability_class" in df.columns:
        for k, v in df["reliability_class"].value_counts().items():
            write_report(f"{k}: {v}")

    # Correlation
    corr = num.corr()
    section("CORRELATION")
    write_report(corr.to_string())

    plt.figure(figsize=(14, 10))
    if HAS_SNS:
        sns.heatmap(corr, cmap="coolwarm", center=0)
    else:
        plt.imshow(corr.values)
        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=90,
            fontsize=6,
        )
        plt.yticks(
            range(len(corr.columns)),
            corr.columns,
            fontsize=6,
        )
    plt.tight_layout()
    plt.savefig(PLOTS / "correlations" / "pearson_heatmap.png", dpi=300)
    plt.close()

    # Per-feature histograms and boxplots
    for c in num.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df[c], bins=30)
        plt.title(c)
        plt.tight_layout()
        plt.savefig(PLOTS / "histograms" / f"{c}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(4, 6))
        plt.boxplot(df[c])
        plt.title(c)
        plt.tight_layout()
        plt.savefig(PLOTS / "boxplots" / f"{c}.png", dpi=300)
        plt.close()

    # Scatter plots
    pairs = [
        ("depth", "estimated_fidelity"),
        ("depth", "reliability_score"),
        ("total_operations", "estimated_fidelity"),
        ("entangling_gates", "reliability_score"),
        ("parameterized_gates", "reliability_score"),
        ("number_of_qubits", "reliability_score"),
    ]
    for x, y in pairs:
        if x in df.columns and y in df.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(df[x], df[y], s=6)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.tight_layout()
            plt.savefig(
                PLOTS / "scatterplots" / f"{x}_vs_{y}.png", dpi=300
            )
            plt.close()

    # Outliers
    section("OUTLIERS")
    for c in num.columns:
        q1, q3 = num[c].quantile(0.25), num[c].quantile(0.75)
        iqr = q3 - q1
        o = ((num[c] < q1 - 1.5 * iqr) | (num[c] > q3 + 1.5 * iqr)).sum()
        write_report(f"{c}: {o}")

    # ML Readiness
    section("ML READINESS")
    write_report("READY FOR MACHINE LEARNING")

    print("EDA complete.")


if __name__ == "__main__":
    run_eda()
