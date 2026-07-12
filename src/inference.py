"""
inference.py
------------
Single Responsibility:
    Given a single OpenQASM 3 (.qasm) file, run it through the existing
    parsing/analysis/feature-extraction pipeline, feed the resulting
    feature vector to the trained model, and report a predicted
    reliability class, an estimated reliability score, and a confidence
    value -- both to the console-facing report file and as a returned,
    programmatically usable result.

This module intentionally does NOT:
    - Simulate the circuit under a noise model (see `noise_simulator.py`)
      -- by design, this module only reuses `parser.py`, `analyzer.py`,
      and `feature_extractor.py`, so a prediction can be produced from
      structural analysis alone, without paying the cost of an Aer
      simulation run. See the "Reliability Score caveat" section below
      for what this trade-off means for the reported score.
    - Preprocess a dataset, train, tune, or evaluate a model (see
      `preprocess.py`, `train_model.py`, `evaluate_model.py`) -- it only
      ever loads the already-fitted winning model and label encoder
      those modules produced.
    - Modify `parser.py`, `analyzer.py`, `feature_extractor.py`,
      `preprocess.py`, or `train_model.py` in any way.
    - Print anything to the console outside of its own
      `if __name__ == "__main__":` CLI summary.

Where this fits in the pipeline:
    Circuit Generator -> Parser -> Analyzer -> Feature Extractor
    -> Noise Simulator -> Dataset Generator -> Preprocessing
    -> Model Training -> Model Evaluation -> Inference (this module)

Reliability Score caveat (read before trusting this number):
    The trained classifier's target, `reliability_class`, was learned
    from features that include real noise-simulation metrics
    (`estimated_fidelity`, `total_variation_distance`,
    `hellinger_distance`, and both success-probability columns -- see
    `dataset_generator.py`'s schema). Because this module deliberately
    does NOT run `noise_simulator.py` on the input circuit, those
    columns cannot be computed here; they are filled with a documented
    placeholder value (`0.0`) so the feature vector still matches the
    shape the model expects. Every such placeholder column is listed
    explicitly in `PredictionResult.unavailable_features` and in the
    generated report, rather than silently pretending the corresponding
    inputs were real.
    Because a numeric regression target was never trained (only the
    `reliability_class` classifier was), "Reliability Score" here is
    NOT a model prediction -- it is a transparent, documented heuristic:
    each class's representative percentage (derived from
    `noise_simulator.RELIABILITY_THRESHOLDS`) is weighted by the
    classifier's own predicted class probabilities. It is reported as
    an *estimate*, and both the report and `PredictionResult` label it
    as such.

    Consistency guarantee: a probability-weighted blend across *all*
    classes' representative percentages can, on its own, land outside
    the predicted class's own threshold band (e.g. predicted class
    MEDIUM but a blended score that reads as LOW under
    `noise_simulator.RELIABILITY_THRESHOLDS`) whenever probability mass
    is spread thinly across classes. Since `reliability_class` is
    always `argmax(class_probabilities)`, the reported score is clamped
    to the predicted class's own band after the probability-weighted
    blend is computed, so the two reported values can never contradict
    each other. See `_estimate_reliability_score`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from src.analyzer import analyze_circuit
from src.feature_extractor import extract_features
from src.noise_simulator import RELIABILITY_THRESHOLDS
from src.parser import QasmParsingError, load_qasm_file

# Bumped whenever this module's feature-construction logic, scoring
# heuristic, or output schema changes in a way future consumers should
# know about. Mirrors `preprocess.PREPROCESS_VERSION` /
# `train_model.TRAIN_MODEL_VERSION` / `evaluate_model.EVALUATE_MODEL_VERSION`.
INFERENCE_VERSION = "1.0.0"

# Default locations, matching every upstream module's own output layout
# exactly -- this module reads those artifacts verbatim.
_DEFAULT_MODELS_DIRECTORY = Path("models")
_DEFAULT_PREPROCESSING_DIRECTORY = _DEFAULT_MODELS_DIRECTORY / "preprocessing"
_DEFAULT_OUTPUT_DIRECTORY = Path("outputs")
_MODEL_METRICS_FILENAME = "model_metrics.pkl"

# Class score bands (0-100 scale), derived directly from
# `noise_simulator.RELIABILITY_THRESHOLDS` rather than duplicated as
# separate hardcoded numbers. This is the single source of truth for
# where each class's band starts/ends -- if RELIABILITY_THRESHOLDS is
# ever recalibrated in noise_simulator.py, this module tracks that
# change automatically instead of silently drifting out of sync.
_HIGH_BAND_MIN = RELIABILITY_THRESHOLDS["HIGH"] * 100  # e.g. 95.0
_MEDIUM_BAND_MIN = RELIABILITY_THRESHOLDS["MEDIUM"] * 100  # e.g. 85.0
_SCORE_BAND_MAX = 100.0
_SCORE_BAND_MIN = 0.0

# Per-class (low, high) score band, used both to derive each class's
# representative midpoint (for the probability-weighted blend) and to
# clamp the final estimate so it can never fall outside the predicted
# class's own band -- see `_estimate_reliability_score`.
_CLASS_SCORE_BANDS: dict[str, tuple[float, float]] = {
    "HIGH": (_HIGH_BAND_MIN, _SCORE_BAND_MAX),
    "MEDIUM": (_MEDIUM_BAND_MIN, _HIGH_BAND_MIN),
    "LOW": (_SCORE_BAND_MIN, _MEDIUM_BAND_MIN),
}

# Representative reliability-score percentage per class, used only for
# the heuristic "Reliability Score" estimate described in the module
# docstring's caveat section. Derived as the midpoint of each class's
# band above, expressed as percentages to match `reliability_score`'s
# own 0-100 scale.
_CLASS_SCORE_MIDPOINTS: dict[str, float] = {
    class_name: (low + high) / 2 for class_name, (low, high) in _CLASS_SCORE_BANDS.items()
}

# Base (non-gate) feature names this module can compute directly from
# `analyzer.analyze_circuit` and `feature_extractor.extract_features`,
# mapped to the column name `dataset_generator.py` used for the same
# quantity when building the training dataset. Kept explicit so the
# mapping between "what these two modules call it" and "what the model
# was trained on" is auditable in one place rather than scattered
# through the feature-building function.
_STRUCTURAL_FEATURE_MAP: dict[str, str] = {
    "number_of_qubits": "num_qubits",
    "number_of_classical_bits": "num_clbits",
    "depth": "depth",
    "width": "width",
    "total_operations": "total_operations",
}
_ML_FEATURE_MAP: dict[str, str] = {
    "single_qubit_gates": "single_qubit_gates",
    "two_qubit_gates": "two_qubit_gates",
    "three_qubit_gates": "three_qubit_gates",
    "measurement_gates": "measurement_gates",
    "parameterized_gates": "parameterized_gates",
    "entangling_gates": "entangling_gates",
}

# Feature columns this module cannot compute without running
# `noise_simulator.py` (deliberately out of scope -- see module
# docstring's caveat). Filled with this placeholder value.
_NOISE_DERIVED_COLUMNS: tuple[str, ...] = (
    "estimated_fidelity",
    "total_variation_distance",
    "hellinger_distance",
    "success_probability_ideal",
    "success_probability_noisy",
)
_NOISE_PLACEHOLDER_VALUE = 0.0


class InferenceError(Exception):
    """Raised when a prediction cannot be produced as configured.

    Kept as a project-specific exception -- mirroring `EvaluationError`
    (evaluate_model.py), `TrainModelError` (train_model.py), and
    `PreprocessingError` (preprocess.py) -- so callers can catch one
    stable error type regardless of which internal step failed (a bad
    QASM file, missing trained-model artifacts, a feature-schema
    mismatch, a write failure).
    """


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for one inference run.

    Attributes:
        models_directory: Directory containing the winning model
            (`<name>.pkl`) and `model_metrics.pkl`, as written by
            `train_model.py`.
        feature_columns_path: Path to the feature column list JSON
            written by `preprocess.py`.
        label_encoder_path: Path to the fitted `LabelEncoder` `.joblib`
            file written by `preprocess.py`.
        output_directory: Directory to write `prediction_report.txt`
            (and a companion `prediction_result.json`) into. Created if
            it doesn't exist.
    """

    models_directory: str | Path = _DEFAULT_MODELS_DIRECTORY
    feature_columns_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "feature_columns.json"
    label_encoder_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "label_encoder.joblib"
    output_directory: str | Path = _DEFAULT_OUTPUT_DIRECTORY


@dataclass
class PredictionResult:
    """The full result of one circuit's inference run.

    Attributes:
        qasm_path: Path to the input `.qasm` file.
        circuit_name: Circuit name (from `analyzer.analyze_circuit`, or
            the file stem if the circuit was unnamed).
        model_name: Name of the model that produced this prediction.
        reliability_class: Predicted class label (LOW / MEDIUM / HIGH).
        confidence: The predicted class's own probability, from the
            model's `predict_proba` output (in [0.0, 1.0]).
        reliability_score_estimate: Heuristic 0-100 score estimate (see
            module docstring's caveat) -- NOT a direct model prediction.
        class_probabilities: Every class's predicted probability,
            keyed by class label.
        unavailable_features: Feature column names that could not be
            computed from parsing/analysis/feature-extraction alone
            and were filled with a placeholder value (see module
            docstring's caveat).
        report_path: Path to the written `prediction_report.txt`.
        result_json_path: Path to the written companion JSON result.
    """

    qasm_path: Path
    circuit_name: str
    model_name: str
    reliability_class: str
    confidence: float
    reliability_score_estimate: float
    class_probabilities: dict[str, float] = field(default_factory=dict)
    unavailable_features: list[str] = field(default_factory=list)
    report_path: Path | None = None
    result_json_path: Path | None = None


# ---------------------------------------------------------------------------
# Loading trained artifacts
# ---------------------------------------------------------------------------


def _load_winning_model_name(model_metrics_path: Path) -> str:
    """Read which candidate won from `train_model.py`'s saved metrics file."""
    if not model_metrics_path.exists():
        raise InferenceError(
            f"Missing model metrics at '{model_metrics_path}'. Run "
            "train_model.py before inference.py."
        )
    try:
        payload = joblib.load(model_metrics_path)
    except (OSError, EOFError) as exc:
        raise InferenceError(
            f"Failed to load model metrics from '{model_metrics_path}': {exc}"
        ) from exc

    winning_model = payload.get("winning_model")
    if not winning_model:
        raise InferenceError(f"'{model_metrics_path}' does not contain a 'winning_model' key.")
    return str(winning_model)


def _load_model(models_directory: Path, model_name: str) -> ClassifierMixin:
    """Load the winning model's fitted estimator from `models/<name>.pkl`."""
    model_path = models_directory / f"{model_name}.pkl"
    if not model_path.exists():
        raise InferenceError(
            f"Winning model file not found at '{model_path}'. Run "
            "train_model.py before inference.py."
        )
    try:
        return joblib.load(model_path)
    except (OSError, EOFError) as exc:
        raise InferenceError(f"Failed to load model from '{model_path}': {exc}") from exc


def _load_feature_columns(path: Path) -> list[str]:
    """Load the ordered feature column list saved by `preprocess.py`."""
    if not path.exists():
        raise InferenceError(f"Missing feature columns file at '{path}'. Run preprocess.py first.")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceError(f"Failed to read feature columns from '{path}': {exc}") from exc


def _load_label_encoder(path: Path) -> LabelEncoder:
    """Load the fitted `LabelEncoder` saved by `preprocess.py`."""
    if not path.exists():
        raise InferenceError(f"Missing label encoder at '{path}'. Run preprocess.py first.")
    try:
        return joblib.load(path)
    except (OSError, EOFError) as exc:
        raise InferenceError(f"Failed to load label encoder from '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Circuit loading and feature construction
# ---------------------------------------------------------------------------


def _load_circuit(qasm_path: Path) -> Any:
    """Parse the input `.qasm` file via `parser.load_qasm_file`.

    Raises:
        InferenceError: If the file is missing, not a `.qasm` file, or
            fails to parse.
    """
    try:
        return load_qasm_file(qasm_path)
    except FileNotFoundError as exc:
        raise InferenceError(f"QASM file not found: {qasm_path}") from exc
    except (ValueError, QasmParsingError) as exc:
        raise InferenceError(f"Invalid QASM file '{qasm_path}': {exc}") from exc


def _build_raw_feature_dict(circuit: Any) -> tuple[dict[str, float], dict[str, int]]:
    """Compute every feature this module CAN derive, via analyzer + feature_extractor.

    Args:
        circuit: A parsed `QuantumCircuit`, as returned by `_load_circuit`.

    Returns:
        A tuple of:
            - A flat dict of structural + ML feature values, keyed by
              the column name `dataset_generator.py` used for the same
              quantity (see `_STRUCTURAL_FEATURE_MAP` / `_ML_FEATURE_MAP`).
            - The circuit's raw gate-count dict (from
              `analyzer.analyze_circuit`), used separately to populate
              `gate_*` columns against the model's own gate vocabulary.
    """
    analysis = analyze_circuit(circuit)
    features = extract_features(circuit)

    raw_features: dict[str, float] = {}
    for target_column, analysis_key in _STRUCTURAL_FEATURE_MAP.items():
        raw_features[target_column] = analysis[analysis_key]
    for target_column, feature_key in _ML_FEATURE_MAP.items():
        raw_features[target_column] = features[feature_key]

    return raw_features, analysis["gate_counts"]


def _build_feature_vector(
    raw_features: dict[str, float],
    gate_counts: dict[str, int],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the model's exact input row, in the trained column order.

    For every column the model expects:
        - Structural / ML feature columns are pulled from `raw_features`.
        - `gate_<name>` columns are pulled from `gate_counts` (0 if this
          circuit never used that gate).
        - Any other column (in practice, only the noise-derived columns
          in `_NOISE_DERIVED_COLUMNS`) is filled with
          `_NOISE_PLACEHOLDER_VALUE` and recorded as unavailable.

    Args:
        raw_features: Output of `_build_raw_feature_dict`'s first element.
        gate_counts: Output of `_build_raw_feature_dict`'s second element.
        feature_columns: The model's expected column order, from
            `feature_columns.json`.

    Returns:
        A tuple of (single-row DataFrame ready for `model.predict`,
        list of column names that had to be filled with a placeholder).
    """
    row: dict[str, float] = {}
    unavailable: list[str] = []

    for column in feature_columns:
        if column in raw_features:
            row[column] = raw_features[column]
        elif column.startswith("gate_"):
            gate_name = column[len("gate_"):]
            row[column] = gate_counts.get(gate_name, 0)
        elif column in _NOISE_DERIVED_COLUMNS:
            row[column] = _NOISE_PLACEHOLDER_VALUE
            unavailable.append(column)
        else:
            # Any other unrecognized column (e.g. a future feature this
            # module doesn't yet know how to compute): fail loudly
            # rather than silently guessing a value that could distort
            # the prediction without anyone noticing.
            raise InferenceError(
                f"Cannot compute required feature column '{column}' from "
                "parser/analyzer/feature_extractor output, and it is not "
                "a recognized noise-derived placeholder column."
            )

    feature_row = pd.DataFrame([row], columns=feature_columns)
    return feature_row, unavailable


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _predict(
    model: ClassifierMixin,
    feature_row: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> tuple[str, float, dict[str, float]]:
    """Run the model on one feature row and decode its class prediction.

    Raises:
        InferenceError: If the model has no `predict_proba` method.
    """
    if not hasattr(model, "predict_proba"):
        raise InferenceError(
            f"Model type '{type(model).__name__}' has no 'predict_proba' method; "
            "cannot compute a confidence score."
        )

    probabilities = model.predict_proba(feature_row)[0]
    class_names = [str(label) for label in label_encoder.classes_]
    class_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(class_names, probabilities)
    }

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, class_probabilities


def _estimate_reliability_score(
    class_probabilities: dict[str, float], predicted_class: str
) -> float:
    """Estimate a 0-100 reliability score from the classifier's class probabilities.

    See the module docstring's "Reliability Score caveat" section for
    why this is a heuristic, not a direct model prediction: it starts as
    the probability-weighted average of each class's representative
    percentage in `_CLASS_SCORE_MIDPOINTS`.

    That raw blend, on its own, is not guaranteed to fall within the
    predicted class's own score band -- e.g. probabilities spread
    roughly evenly across MEDIUM and HIGH (with some LOW) can blend down
    below the MEDIUM band's floor even though MEDIUM is the argmax
    class. Since `reliability_class` (the caller's `predicted_class`)
    is always that argmax, the two numbers must agree: this function
    clamps the raw blend into `predicted_class`'s own
    `_CLASS_SCORE_BANDS` range before returning it.

    Args:
        class_probabilities: Per-class probabilities from `_predict`.
        predicted_class: The class label `_predict` selected as the
            argmax of `class_probabilities` -- the score is clamped to
            stay consistent with this class.

    Returns:
        A float in [0.0, 100.0], guaranteed to fall within
        `predicted_class`'s own score band.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    for class_name, probability in class_probabilities.items():
        midpoint = _CLASS_SCORE_MIDPOINTS.get(class_name)
        if midpoint is None:
            continue  # Unrecognized class label; skip rather than guess.
        weighted_sum += probability * midpoint
        total_weight += probability

    raw_estimate = 0.0 if total_weight == 0.0 else weighted_sum / total_weight

    band = _CLASS_SCORE_BANDS.get(predicted_class)
    if band is None:
        # Unrecognized predicted class label; nothing to clamp against,
        # return the raw blend rather than guessing a band.
        return raw_estimate

    band_low, band_high = band
    return min(max(raw_estimate, band_low), band_high)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _resolve_circuit_display_name(circuit: Any, qasm_path: Path) -> str:
    """Return the circuit's explicit name, or the file stem if unnamed."""
    analysis = analyze_circuit(circuit)
    return analysis["name"] or qasm_path.stem


def _write_prediction_report(
    path: Path,
    result: PredictionResult,
) -> None:
    """Write the plain-text prediction report.

    Raises:
        InferenceError: If the file cannot be written.
    """
    separator = "=" * 60
    lines: list[str] = [
        separator,
        "Quantum-Reliability-AI -- Prediction Report",
        separator,
        "",
        f"Input Circuit      : {result.qasm_path}",
        f"Circuit Name       : {result.circuit_name}",
        f"Model Used         : {result.model_name}",
        f"Generated At (UTC) : {datetime.now(timezone.utc).isoformat()}",
        "",
        "-" * 60,
        "Prediction",
        "-" * 60,
        f"Reliability Class          : {result.reliability_class}",
        f"Confidence                 : {result.confidence:.4f} ({result.confidence * 100:.1f}%)",
        f"Reliability Score (est.)   : {result.reliability_score_estimate:.2f} / 100",
        "",
        "Class Probabilities:",
    ]
    for class_name, probability in sorted(
        result.class_probabilities.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"  {class_name:<10}: {probability:.4f} ({probability * 100:.1f}%)")

    lines += [
        "",
        "-" * 60,
        "Notes and Limitations",
        "-" * 60,
        (
            "Reliability Class and Confidence are direct outputs of the "
            f"trained '{result.model_name}' classifier."
        ),
        (
            "Reliability Score is NOT a direct model prediction -- no "
            "regression model was trained for it. It is a heuristic "
            "estimate: each class's representative percentage, weighted "
            "by the classifier's own predicted class probabilities."
        ),
    ]

    if result.unavailable_features:
        lines += [
            (
                "This module reuses only parser.py, analyzer.py, and "
                "feature_extractor.py -- it does not run noise_simulator.py "
                "on the input circuit. The following model input feature(s) "
                "could therefore not be computed and were filled with a "
                f"placeholder value ({_NOISE_PLACEHOLDER_VALUE}), which may "
                "reduce prediction accuracy relative to the model's "
                "reported test-set performance:"
            )
        ]
        for feature_name in result.unavailable_features:
            lines.append(f"  - {feature_name}")

    lines.append("")
    lines.append("Prediction Complete.")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text("\n".join(lines) + "\n")
    except OSError as exc:
        raise InferenceError(f"Failed to write prediction report to '{path}': {exc}") from exc


def _write_prediction_json(path: Path, result: PredictionResult) -> None:
    """Write a machine-readable companion to the text report.

    A bonus artifact (not strictly required) mirroring the project's
    existing convention of pairing a human-readable report with a
    structured metadata file (see `preprocessing_metadata.json`,
    `training_metadata.json`).

    Raises:
        InferenceError: If the file cannot be written.
    """
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inference_version": INFERENCE_VERSION,
        "qasm_path": str(result.qasm_path),
        "circuit_name": result.circuit_name,
        "model_name": result.model_name,
        "reliability_class": result.reliability_class,
        "confidence": result.confidence,
        "reliability_score_estimate": result.reliability_score_estimate,
        "class_probabilities": result.class_probabilities,
        "unavailable_features": result.unavailable_features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        raise InferenceError(f"Failed to write prediction JSON to '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_inference(
    qasm_path: str | Path,
    config: InferenceConfig | None = None,
) -> PredictionResult:
    """Predict reliability class, score, and confidence for one QASM circuit.

    End-to-end pipeline:
        1. Parse the `.qasm` file via `parser.load_qasm_file`.
        2. Run `analyzer.analyze_circuit` and
           `feature_extractor.extract_features` on the parsed circuit.
        3. Assemble a feature vector in the trained model's exact
           column order, filling any noise-derived columns this module
           cannot compute with a documented placeholder (see module
           docstring's caveat).
        4. Load the winning model (identified via `train_model.py`'s
           `model_metrics.pkl`) and the fitted label encoder, and
           predict the reliability class, its confidence, and a
           heuristic reliability score estimate.
        5. Write `prediction_report.txt` (and a companion
           `prediction_result.json`) under `config.output_directory`.

    Args:
        qasm_path: Path to the `.qasm` file to analyze and predict on.
        config: Inference configuration. Defaults to
            `InferenceConfig()` (reads from `models/` and
            `models/preprocessing/`, writes to `outputs/`).

    Returns:
        A `PredictionResult` with the prediction, confidence, score
        estimate, and every output file's path.

    Raises:
        InferenceError: If the QASM file cannot be parsed, required
            trained-model artifacts are missing, the feature vector
            cannot be assembled, or any output file cannot be written.
    """
    active_config = config or InferenceConfig()
    resolved_qasm_path = Path(qasm_path)
    models_directory = Path(active_config.models_directory)
    output_directory = Path(active_config.output_directory)

    circuit = _load_circuit(resolved_qasm_path)
    circuit_name = _resolve_circuit_display_name(circuit, resolved_qasm_path)

    raw_features, gate_counts = _build_raw_feature_dict(circuit)
    feature_columns = _load_feature_columns(Path(active_config.feature_columns_path))
    feature_row, unavailable_features = _build_feature_vector(
        raw_features, gate_counts, feature_columns
    )

    model_name = _load_winning_model_name(models_directory / _MODEL_METRICS_FILENAME)
    model = _load_model(models_directory, model_name)
    label_encoder = _load_label_encoder(Path(active_config.label_encoder_path))

    reliability_class, confidence, class_probabilities = _predict(
        model, feature_row, label_encoder
    )
    reliability_score_estimate = _estimate_reliability_score(
        class_probabilities, reliability_class
    )

    result = PredictionResult(
        qasm_path=resolved_qasm_path,
        circuit_name=circuit_name,
        model_name=model_name,
        reliability_class=reliability_class,
        confidence=confidence,
        reliability_score_estimate=reliability_score_estimate,
        class_probabilities=class_probabilities,
        unavailable_features=unavailable_features,
    )

    report_path = output_directory / "prediction_report.txt"
    result_json_path = output_directory / "prediction_result.json"
    _write_prediction_report(report_path, result)
    _write_prediction_json(result_json_path, result)

    result.report_path = report_path
    result.result_json_path = result_json_path
    return result


def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the `__main__` CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Predict circuit reliability class, score, and confidence "
        "for a single OpenQASM 3 file."
    )
    parser.add_argument("qasm_path", help="Path to the .qasm file to analyze.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Usage:
        python -m src.inference path/to/circuit.qasm
    """
    args = _parse_cli_args(argv if argv is not None else sys.argv[1:])

    try:
        result = run_inference(args.qasm_path)
    except InferenceError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("Prediction Complete")
    print("=" * 60)
    print(f"Circuit                  : {result.circuit_name}")
    print(f"Model Used               : {result.model_name}")
    print(f"Reliability Class        : {result.reliability_class}")
    print(f"Confidence               : {result.confidence:.4f} ({result.confidence * 100:.1f}%)")
    print(f"Reliability Score (est.) : {result.reliability_score_estimate:.2f} / 100")
    if result.unavailable_features:
        print(
            f"Note: {len(result.unavailable_features)} noise-derived feature(s) "
            "were unavailable and filled with a placeholder -- see the report "
            "for details."
        )
    print()
    print(f"Report : {result.report_path}")
    print(f"JSON   : {result.result_json_path}")


if __name__ == "__main__":
    main()
