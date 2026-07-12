"""
main.py
-------
Entry point and presentation layer for Quantum-Reliability-AI.

Responsibilities:
    - Load a .qasm file path (from a CLI argument)
    - Delegate parsing to src.parser
    - Delegate structural analysis to src.analyzer
    - Delegate ML-ready feature extraction to src.feature_extractor
    - Delegate ideal-vs-noisy simulation to src.noise_simulator
    - Format and print a structural report, a feature-extraction report,
      and a noise-simulation report to the terminal

This is the ONLY module in the project allowed to print to the console.
Keeping I/O confined here means `parser.py`, `analyzer.py`,
`feature_extractor.py`, and `noise_simulator.py` stay reusable by future
non-CLI consumers (a REST API, a dashboard, a batch runner, a future
dataset generator) without modification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from src.analyzer import analyze_circuit
from src.feature_extractor import extract_features
from src.noise_simulator import CircuitSimulationError, simulate_noise
from src.parser import QasmParsingError, load_qasm_file

REPORT_WIDTH = 50
FEATURE_REPORT_WIDTH = 30
NOISE_REPORT_WIDTH = 30


def build_report_lines(file_name: str, analysis: dict[str, Any]) -> list[str]:
    """Format analysis results into printable report lines.

    Args:
        file_name: Original .qasm file name, used as a display fallback.
        analysis: The dictionary returned by `analyze_circuit`.

    Returns:
        A list of strings, each representing one printable line.
    """
    separator = "=" * REPORT_WIDTH
    display_name = analysis["name"] or file_name

    lines = [
        separator,
        "Quantum Circuit Analysis Report",
        separator,
        f"{'File Name':<17}: {display_name}",
        f"{'Qubits':<17}: {analysis['num_qubits']}",
        f"{'Classical Bits':<17}: {analysis['num_clbits']}",
        f"{'Circuit Depth':<17}: {analysis['depth']}",
        f"{'Circuit Width':<17}: {analysis['width']}",
        f"{'Total Operations':<17}: {analysis['total_operations']}",
        "",
        "Gate Statistics",
    ]

    for gate_name, count in analysis["gate_counts"].items():
        lines.append(f"{gate_name.upper():<10} : {count}")

    lines.append(separator)
    return lines


def build_feature_report_lines(features: dict[str, Any]) -> list[str]:
    """Format extracted features into printable report lines.

    Args:
        features: The dictionary returned by `extract_features`.

    Returns:
        A list of strings, each representing one printable line.
    """
    separator = "=" * FEATURE_REPORT_WIDTH

    lines = [
        separator,
        "Feature Extraction Report",
        separator,
        "",
        f"Single-Qubit Gates : {features['single_qubit_gates']}",
        "",
        f"Two-Qubit Gates : {features['two_qubit_gates']}",
        "",
        f"Three-Qubit Gates : {features['three_qubit_gates']}",
        "",
        f"Parameterized Gates : {features['parameterized_gates']}",
        "",
        f"Entangling Gates : {features['entangling_gates']}",
        "",
        f"Measurement Gates : {features['measurement_gates']}",
        "",
        "Feature Dictionary",
        "",
        json.dumps(features, indent=4),
    ]
    return lines


def build_noise_report_lines(noise_result: dict[str, Any]) -> list[str]:
    """Format noise simulation results into printable report lines.

    Args:
        noise_result: The dictionary returned by `simulate_noise`.

    Returns:
        A list of strings, each representing one printable line.
    """
    separator = "=" * NOISE_REPORT_WIDTH

    lines = [
        separator,
        "Circuit Reliability Report",
        separator,
        "",
        "Circuit Reliability",
        noise_result["circuit_reliability"],
        "",
        "Estimated Reliability",
        f"{noise_result['estimated_reliability_percent']:.1f}%",
        "",
        "Noise Assessment",
        noise_result["noise_assessment"],
        "",
        noise_result["reliability_explanation"],
        noise_result["noise_assessment_explanation"],
        "",
        "-" * NOISE_REPORT_WIDTH,
        "Detailed Metrics",
        "-" * NOISE_REPORT_WIDTH,
        "",
        "Shots",
        str(noise_result["total_shots"]),
        "",
        "Estimated Fidelity",
        f"{noise_result['estimated_fidelity']:.3f}",
        "",
        "Total Variation Distance (Noise Impact)",
        f"{noise_result['total_variation_distance']:.3f}",
        "",
        "Hellinger Distance",
        f"{noise_result['hellinger_distance']:.3f}",
        "",
        "Ideal Success Probability",
        f"{noise_result['ideal_success_probability']:.3f}",
        "",
        "Noisy Success Probability",
        f"{noise_result['noisy_success_probability']:.3f}",
        "",
        f"Shot-Noise Margin of Error: \u00b1{noise_result['success_probability_margin_of_error']:.3f}",
        noise_result["shot_noise_note"],
        "",
        "Ideal Counts",
        json.dumps(noise_result["ideal_counts"], indent=4),
        "",
        "Noisy Counts",
        json.dumps(noise_result["noisy_counts"], indent=4),
    ]
    return lines


def run(qasm_path: str) -> None:
    """Load, analyze, and print a report for the given .qasm file.

    Args:
        qasm_path: Path to the .qasm file to analyze.
    """
    file_name = Path(qasm_path).name

    try:
        circuit = load_qasm_file(qasm_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return
    except (ValueError, QasmParsingError) as exc:
        print(f"Error: Invalid QASM file. {exc}")
        return

    analysis = analyze_circuit(circuit)

    for line in build_report_lines(file_name, analysis):
        print(line)

    features = extract_features(circuit)
    print()

    for line in build_feature_report_lines(features):
        print(line)

    print()

    try:
        noise_result = simulate_noise(circuit)
    except CircuitSimulationError as exc:
        print(f"Error: Noise simulation failed. {exc}")
        return

    for line in build_noise_report_lines(noise_result):
        print(line)


def main() -> None:
    """CLI entry point.

    Usage:
        python main.py path/to/circuit.qasm
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_qasm_file>")
        sys.exit(1)

    run(sys.argv[1])


if __name__ == "__main__":
    main()
