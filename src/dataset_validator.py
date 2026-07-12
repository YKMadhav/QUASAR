"""
dataset_validator.py
--------------------
Validates the generated training dataset before machine learning.

Run:
    python -m src.dataset_validator
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_PATH = Path("datasets/training_dataset.csv")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "validation_report.txt"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def write_report(text: str, mode: str = "a") -> None:
    """Append *text* to the validation report file."""
    with open(REPORT_PATH, mode, encoding="utf-8") as f:
        f.write(text + "\n")


def section(title: str) -> None:
    """Write a section header to the report."""
    line = "=" * 70
    write_report("\n" + line)
    write_report(title)
    write_report(line)


def report_check(name: str, invalid_count: int) -> None:
    """Write a PASS/FAIL consistency check result."""
    if invalid_count == 0:
        write_report(f"[PASS] {name}")
    else:
        write_report(
            f"[FAIL] {name} -> {invalid_count} inconsistent rows"
        )


# ---------------------------------------------------------
# Core validation
# ---------------------------------------------------------


def validate() -> None:
    """Run the full dataset validation pipeline."""

    REPORT_DIR.mkdir(exist_ok=True)

    print("\nLoading dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")

    REPORT_PATH.write_text("")

    # Dataset Overview
    section("DATASET OVERVIEW")
    rows, cols = df.shape
    write_report(f"Rows    : {rows}")
    write_report(f"Columns : {cols}")
    write_report("\nColumn Names:")
    for column in df.columns:
        write_report(f"  - {column}")

    # Data Types
    section("COLUMN DATA TYPES")
    for col, dtype in df.dtypes.items():
        write_report(f"{col:<35} {dtype}")

    # Missing Values
    section("MISSING VALUES")
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    write_report(f"Total Missing Values : {total_missing}\n")
    for col, value in missing.items():
        if value > 0:
            write_report(f"{col:<35} {value}")
    if total_missing == 0:
        write_report("No missing values found.")

    # Duplicate Rows
    section("DUPLICATE ROWS")
    duplicates = int(df.duplicated().sum())
    write_report(f"Duplicate Rows : {duplicates}")

    # Duplicate Circuits
    section("DUPLICATE CIRCUITS")
    if "circuit_name" in df.columns:
        duplicate_circuits = int(
            df["circuit_name"].duplicated().sum()
        )
        write_report(
            f"Duplicate Circuit Names : {duplicate_circuits}"
        )
    else:
        write_report("Column 'circuit_name' not found.")

    # Numeric Summary
    section("NUMERIC FEATURE SUMMARY")
    numeric_df = df.select_dtypes(include=np.number)
    summary = numeric_df.describe().T
    write_report(summary.to_string())

    # Reliability Score
    if "reliability_score" in df.columns:
        section("RELIABILITY SCORE")
        score = df["reliability_score"]
        write_report(f"Minimum : {score.min():.4f}")
        write_report(f"Maximum : {score.max():.4f}")
        write_report(f"Mean    : {score.mean():.4f}")
        write_report(f"Median  : {score.median():.4f}")

    # Reliability Class
    if "reliability_class" in df.columns:
        section("RELIABILITY CLASS DISTRIBUTION")
        counts = df["reliability_class"].value_counts()
        for label, count in counts.items():
            percentage = (count / len(df)) * 100
            write_report(
                f"{label:<15} {count:>7} ({percentage:.2f}%)"
            )

    # Invalid Values
    section("INVALID VALUE CHECKS")
    checks = {
        "estimated_fidelity": lambda x: (x < 0) | (x > 1),
        "success_probability_ideal": lambda x: (x < 0) | (x > 1),
        "success_probability_noisy": lambda x: (x < 0) | (x > 1),
        "total_variation_distance": lambda x: x < 0,
        "hellinger_distance": lambda x: x < 0,
        "depth": lambda x: x < 0,
        "width": lambda x: x < 0,
        "number_of_qubits": lambda x: x <= 0,
        "total_operations": lambda x: x < 0,
    }
    for column, rule in checks.items():
        if column in df.columns:
            invalid = int(rule(df[column]).sum())
            write_report(f"{column:<35} {invalid}")

    # Internal Consistency Checks
    section("INTERNAL CONSISTENCY CHECKS")

    if (
        "number_of_classical_bits" in df.columns
        and "measurement_gates" in df.columns
    ):
        invalid = (
            df["number_of_classical_bits"]
            != df["measurement_gates"]
        ).sum()
        report_check(
            "Classical bits equal measurement gates", int(invalid)
        )

    required_ops = [
        "single_qubit_gates",
        "two_qubit_gates",
        "three_qubit_gates",
        "measurement_gates",
        "total_operations",
    ]
    if all(col in df.columns for col in required_ops):
        calculated = (
            df["single_qubit_gates"]
            + df["two_qubit_gates"]
            + df["three_qubit_gates"]
            + df["measurement_gates"]
        )
        invalid = (calculated != df["total_operations"]).sum()
        report_check(
            "Total operations match gate-category totals",
            int(invalid),
        )

    single_gate_columns = [
        "gate_h", "gate_x", "gate_y", "gate_z",
        "gate_s", "gate_sdg", "gate_sx", "gate_t",
        "gate_rx", "gate_ry", "gate_rz",
    ]
    if (
        all(col in df.columns for col in single_gate_columns)
        and "single_qubit_gates" in df.columns
    ):
        calculated = df[single_gate_columns].sum(axis=1)
        invalid = (calculated != df["single_qubit_gates"]).sum()
        report_check("Single-qubit gate totals", int(invalid))

    two_gate_columns = [
        "gate_cx", "gate_cz", "gate_cp", "gate_swap",
    ]
    if (
        all(col in df.columns for col in two_gate_columns)
        and "two_qubit_gates" in df.columns
    ):
        calculated = df[two_gate_columns].sum(axis=1)
        invalid = (calculated != df["two_qubit_gates"]).sum()
        report_check("Two-qubit gate totals", int(invalid))

    if (
        "gate_ccx" in df.columns
        and "three_qubit_gates" in df.columns
    ):
        invalid = (
            df["gate_ccx"] != df["three_qubit_gates"]
        ).sum()
        report_check("Three-qubit gate totals", int(invalid))

    entangling_cols = [
        "gate_cx", "gate_cz", "gate_cp", "gate_swap", "gate_ccx",
        "entangling_gates",
    ]
    if all(col in df.columns for col in entangling_cols):
        calculated = (
            df["gate_cx"]
            + df["gate_cz"]
            + df["gate_cp"]
            + df["gate_swap"]
            + df["gate_ccx"]
        )
        invalid = (calculated != df["entangling_gates"]).sum()
        report_check("Entangling gate totals", int(invalid))

    if (
        "gate_measure" in df.columns
        and "measurement_gates" in df.columns
    ):
        invalid = (
            df["gate_measure"] != df["measurement_gates"]
        ).sum()
        report_check("Measurement gate totals", int(invalid))

    # Final Summary
    section("VALIDATION SUMMARY")
    write_report(f"Rows Checked              : {rows}")
    write_report(f"Columns Checked           : {cols}")
    write_report(f"Missing Values            : {total_missing}")
    write_report(f"Duplicate Rows            : {duplicates}")
    write_report("\nValidation Complete.")

    print("\nValidation completed successfully.")
    print(f"Report saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    validate()
