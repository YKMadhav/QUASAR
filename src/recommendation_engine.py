"""
recommendation_engine.py
--------------------------
Single Responsibility:
    Given a single OpenQASM 3 (.qasm) circuit, combine `inference.py`-style
    predicted reliability class/score with the SHAP feature-importance
    rankings and LIME local explanations `explainability.py` has already
    computed for the winning model, to produce a ranked list of concrete,
    actionable structural recommendations for improving that circuit's
    predicted reliability -- e.g. reduce CX gates, reduce circuit depth,
    reduce entangling gates, replace noisy gate sequences with cheaper
    equivalents, reduce the number of measurements -- and write a single
    human-readable `recommendation_report.txt`.

This module intentionally does NOT:
    - Simulate the circuit under a noise model itself (see
      `noise_simulator.py`) -- like `inference.py`, it reuses only
      `parser.py`, `analyzer.py`, and `feature_extractor.py` plus the
      already-trained model, so a recommendation can be produced
      cheaply, without paying for a fresh Aer run.
    - Compute SHAP values itself, in any form. `explainability.py` is
      the ONLY module in this project that ever constructs a SHAP
      explainer (it already knows how to safely handle
      `GradientBoostingClassifier` and any other classifier type, via
      its `TreeExplainer` -> model-agnostic `shap.Explainer` fallback
      strategy -- see that module's docstring). This module never
      builds a `shap.TreeExplainer` or `shap.Explainer` of its own; it
      always calls `explainability.explain_model` and consumes the
      `ExplainabilitySummary` that comes back.
    - Modify the input circuit or emit a rewritten `.qasm` file --
      recommendations are advisory text, not automatic circuit
      transformation. Applying them is left to a human or a future
      `circuit_optimizer.py`.
    - Train, retrain, or evaluate any model (see `train_model.py`,
      `evaluate_model.py`).
    - Print anything to the console outside of its own
      `if __name__ == "__main__":` CLI summary.

Where this fits in the pipeline:
    Circuit Generator -> Parser -> Analyzer -> Feature Extractor
    -> Noise Simulator -> Dataset Generator -> Preprocessing
    -> Model Training -> Model Evaluation -> Explainability
    -> Inference -> Recommendation Engine (this module)

Design summary:
    - Reuses `inference.py`'s own artifact-loading and feature-vector
      construction logic conceptually (winning model via
      `model_metrics.pkl`, feature columns, label encoder) so this
      module's view of "what predicts reliability" never drifts from
      what `inference.py` already reports to a user. Predicting this
      one circuit's reliability class/confidence is plain model
      inference (`model.predict_proba`), not SHAP, so it still happens
      locally in this module -- see `_predict`.
    - Explanation context comes entirely from `explainability.py`:
      `_get_explainability_summary` calls `explainability.explain_model`
      (or reuses an already-computed `ExplainabilitySummary` a caller
      passes in) to obtain that module's per-class SHAP feature
      rankings, its representative LIME local explanations, and the
      path to its written `explanation_report.txt`. No SHAP computation
      of any kind is duplicated here.
    - `_dataset_level_shap_magnitude` turns the predicted class's own
      SHAP ranking (or, for a non-LOW prediction, the LOW class's own
      ranking -- the model's clearest dataset-wide signal for "what
      drives an unreliable prediction") into a feature -> magnitude
      lookup. Combined with a matching `LimeExample` (the representative
      circuit LIME already explained for this same predicted class),
      this stands in for the single-circuit SHAP attribution the old
      implementation used to recompute from scratch.
    - A small, explicit rule set (`_RECOMMENDATION_RULES`) maps
      structural feature categories (CX-heavy, deep, entangling-heavy,
      T/Toffoli-heavy, over-measured, ...) to plain-language
      recommendations. A rule fires only when both (a) the relevant
      feature(s) are structurally significant for this circuit (above a
      configurable threshold) AND (b) explainability.py's own SHAP
      ranking attributes them a meaningful magnitude toward the
      low-reliability class -- so recommendations stay grounded in the
      model's own stated reasoning, not just raw gate counts.
    - Recommendations are ranked by that SHAP-ranking magnitude, so the
      report leads with whatever the model itself says matters most for
      unreliable predictions among this circuit's structurally
      significant features.
    - Gate-substitution advice (e.g. CCX -> relative-phase Toffoli,
      SWAP -> wire relabeling) is deliberately conservative and
      qualitative: this module never guarantees a substitution preserves
      circuit semantics for arbitrary programs, and always states that
      caveat in the emitted text.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from src.analyzer import analyze_circuit
from src.explainability import (
    ExplainabilityError,
    ExplainabilitySummary,
    ExplainConfig,
    LimeExample,
    explain_model,
)
from src.feature_extractor import extract_features
from src.parser import QasmParsingError, load_qasm_file

# Bumped whenever this module's rule set, scoring, or output schema
# changes in a way future consumers should know about. Mirrors
# `inference.INFERENCE_VERSION` / `explainability.EXPLAINABILITY_VERSION`.
# 2.0.0: this module no longer computes SHAP itself (it used to run its
# own `shap.TreeExplainer` against the winning model for a single
# circuit's feature row). It now calls `explainability.explain_model`
# and reuses that module's own SHAP feature rankings, LIME local
# explanations, and written explanation report -- see the module
# docstring's "Design summary" for why, and `_get_explainability_summary`
# for how.
# 2.1.0: `_dataset_level_shap_magnitude` now reads
# `explainability.ExplainabilitySummary`'s untruncated
# `per_class_feature_ranking_full` / `global_feature_ranking_full`
# instead of the `top_n_features`-truncated display fields those same
# objects also carry. The truncated fields could silently read as 0.0
# for a rule's feature that simply fell outside the top-N display
# cutoff, causing that rule to be incorrectly suppressed as if the
# model assigned it zero importance. Requires explainability.py >= 2.2.0.
RECOMMENDATION_ENGINE_VERSION = "2.1.0"

_DEFAULT_MODELS_DIRECTORY = Path("models")
_DEFAULT_PREPROCESSING_DIRECTORY = _DEFAULT_MODELS_DIRECTORY / "preprocessing"
_DEFAULT_SPLITS_DIRECTORY = _DEFAULT_PREPROCESSING_DIRECTORY / "splits"
_DEFAULT_OUTPUT_DIRECTORY = Path("outputs")
_DEFAULT_EXPLAINABILITY_PLOTS_DIRECTORY = Path("plots") / "explainability"
_DEFAULT_EXPLANATION_REPORT_PATH = Path("reports") / "explanation_report.txt"
_MODEL_METRICS_FILENAME = "model_metrics.pkl"

# The reliability class whose SHAP ranking is used as the "what drives
# an unreliable prediction" signal (see `_dataset_level_shap_magnitude`).
# Matches the class name `preprocess.py` / `train_model.py` produce for
# the lowest reliability tier.
_LOW_RELIABILITY_CLASS_NAME = "LOW"

# Same mapping `inference.py` uses to translate parser/analyzer/
# feature_extractor output into the training dataset's own column
# names. Kept identical (not imported) so this module has no runtime
# dependency on inference.py's internals -- only on the same upstream
# modules it, in turn, also depends on.
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
_NOISE_DERIVED_COLUMNS: tuple[str, ...] = (
    "estimated_fidelity",
    "total_variation_distance",
    "hellinger_distance",
    "success_probability_ideal",
    "success_probability_noisy",
)
_NOISE_PLACEHOLDER_VALUE = 0.0

# Minimum structural thresholds a feature must clear (on the raw
# circuit, not a normalized score) before a rule is even a candidate to
# fire. Deliberately conservative, hand-picked starting points for a v1
# prototype -- not derived from the trained model -- since "is this gate
# count structurally worth mentioning at all" is a domain judgment, not
# something SHAP alone answers.
_MIN_CX_GATES_FOR_FLAG = 5
_MIN_DEPTH_FOR_FLAG = 20
_MIN_ENTANGLING_GATES_FOR_FLAG = 8
_MIN_THREE_QUBIT_GATES_FOR_FLAG = 2
_MIN_MEASUREMENTS_FOR_FLAG = 4
_MIN_SWAP_GATES_FOR_FLAG = 2

# A SHAP contribution is only considered "meaningfully negative" (i.e.
# pushing away from the model's most-reliable class / toward a lower
# reliability_score) if it clears this magnitude. Prevents the engine
# from citing a feature whose attribution is essentially noise.
_MIN_SHAP_MAGNITUDE = 1e-4


class RecommendationEngineError(Exception):
    """Raised when a recommendation cannot be produced as configured.

    Kept as a project-specific exception -- mirroring `InferenceError`
    (inference.py) and `ExplainabilityError` (explainability.py) -- so
    callers can catch one stable error type regardless of which internal
    step failed (a bad QASM file, missing trained-model artifacts, a
    SHAP failure, a write failure).
    """


@dataclass(frozen=True)
class RecommendationConfig:
    """Configuration for one recommendation run.

    Attributes:
        models_directory: Directory containing the winning model
            (`<name>.pkl`) and `model_metrics.pkl`, as written by
            `train_model.py`.
        feature_columns_path: Path to the feature column list JSON
            written by `preprocess.py`.
        label_encoder_path: Path to the fitted `LabelEncoder` `.joblib`
            file written by `preprocess.py`.
        splits_directory: Directory containing `X_test.csv` / `y_test.csv`,
            as written by `preprocess.py`. Passed straight through to
            `explainability.ExplainConfig` -- this module never reads
            the test split itself, `explainability.py` does, when it
            computes the SHAP rankings and LIME examples this module
            consumes (see `_get_explainability_summary`).
        explainability_plots_directory: Directory `explainability.py`
            saves its SHAP/LIME figures into, passed through to
            `explainability.ExplainConfig` when this module has to
            invoke `explain_model` itself (i.e. no precomputed
            `ExplainabilitySummary` was supplied to
            `generate_recommendations`).
        explainability_report_path: Path `explainability.py` writes its
            `explanation_report.txt` to, passed through the same way,
            and also recorded on `RecommendationResult` so a caller can
            point a user at the full dataset-wide explanation alongside
            this circuit's own recommendations.
        output_directory: Directory to write `recommendation_report.txt`
            (and a companion `recommendation_result.json`) into. Created
            if it doesn't exist.
        max_recommendations: Maximum number of recommendations to include
            in the report, after ranking by SHAP ranking magnitude.
    """

    models_directory: str | Path = _DEFAULT_MODELS_DIRECTORY
    feature_columns_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "feature_columns.json"
    label_encoder_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "label_encoder.joblib"
    splits_directory: str | Path = _DEFAULT_SPLITS_DIRECTORY
    explainability_plots_directory: str | Path = _DEFAULT_EXPLAINABILITY_PLOTS_DIRECTORY
    explainability_report_path: str | Path = _DEFAULT_EXPLANATION_REPORT_PATH
    output_directory: str | Path = _DEFAULT_OUTPUT_DIRECTORY
    max_recommendations: int = 5


@dataclass
class Recommendation:
    """One actionable, ranked recommendation for improving a circuit.

    Attributes:
        title: Short, plain-language summary (e.g. "Reduce CX gate count").
        detail: Full explanation of the issue and a concrete suggestion.
        triggering_features: Feature name(s) whose values and SHAP
            ranking caused this recommendation to fire.
        shap_impact: The summed SHAP-ranking magnitude of
            `triggering_features`, reused from `explainability.py`'s
            own per-class ranking (see `_dataset_level_shap_magnitude`)
            rather than recomputed here -- more positive means "the
            model attributes more importance to this feature group in
            driving low-reliability predictions", and drives the
            recommendation's rank.
        category: Short machine-readable tag, one of: "cx_gates",
            "depth", "entangling_gates", "gate_substitution",
            "measurements".
    """

    title: str
    detail: str
    triggering_features: list[str]
    shap_impact: float
    category: str


@dataclass
class RecommendationResult:
    """The full result of one circuit's recommendation run.

    Attributes:
        qasm_path: Path to the input `.qasm` file.
        circuit_name: Circuit name (from `analyzer.analyze_circuit`, or
            the file stem if the circuit was unnamed).
        model_name: Name of the model the recommendations are based on.
        predicted_class: The model's predicted reliability class.
        confidence: The predicted class's own probability.
        recommendations: Ranked list of `Recommendation` objects (most
            impactful first), truncated to `config.max_recommendations`.
        unavailable_features: Feature column names that could not be
            computed from parsing/analysis/feature-extraction alone and
            were filled with a placeholder (same caveat as
            `inference.py` -- noise-derived columns are unavailable
            without running `noise_simulator.py`).
        shap_method: Which SHAP strategy `explainability.py` actually
            used to produce the rankings this run consumed (its
            `ExplainabilitySummary.shap_method`) -- carried through so
            this report is honest about provenance without recomputing
            anything.
        explanation_report_path: Path to `explainability.py`'s own
            `explanation_report.txt`, so a reader can go from this
            circuit-specific report to the full dataset-wide SHAP/LIME
            analysis it was derived from.
        lime_example: The `explainability.py` `LimeExample` (a
            representative test-set circuit LIME already explained) for
            this circuit's predicted class, if one was available. Reused
            as-is, not recomputed for this specific circuit.
        report_path: Path to the written `recommendation_report.txt`.
        result_json_path: Path to the written companion JSON result.
    """

    qasm_path: Path
    circuit_name: str
    model_name: str
    predicted_class: str
    confidence: float
    recommendations: list[Recommendation] = field(default_factory=list)
    unavailable_features: list[str] = field(default_factory=list)
    shap_method: str | None = None
    explanation_report_path: Path | None = None
    lime_example: LimeExample | None = None
    report_path: Path | None = None
    result_json_path: Path | None = None


# ---------------------------------------------------------------------------
# Loading trained artifacts (mirrors inference.py / explainability.py)
# ---------------------------------------------------------------------------


def _load_winning_model_name(model_metrics_path: Path) -> str:
    """Read which candidate won from `train_model.py`'s saved metrics file."""
    if not model_metrics_path.exists():
        raise RecommendationEngineError(
            f"Missing model metrics at '{model_metrics_path}'. Run "
            "train_model.py before recommendation_engine.py."
        )
    try:
        payload = joblib.load(model_metrics_path)
    except (OSError, EOFError) as exc:
        raise RecommendationEngineError(
            f"Failed to load model metrics from '{model_metrics_path}': {exc}"
        ) from exc

    winning_model = payload.get("winning_model")
    if not winning_model:
        raise RecommendationEngineError(
            f"'{model_metrics_path}' does not contain a 'winning_model' key."
        )
    return str(winning_model)


def _load_model(models_directory: Path, model_name: str) -> ClassifierMixin:
    """Load the winning model's fitted estimator from `models/<name>.pkl`."""
    model_path = models_directory / f"{model_name}.pkl"
    if not model_path.exists():
        raise RecommendationEngineError(
            f"Winning model file not found at '{model_path}'. Run "
            "train_model.py before recommendation_engine.py."
        )
    try:
        return joblib.load(model_path)
    except (OSError, EOFError) as exc:
        raise RecommendationEngineError(f"Failed to load model from '{model_path}': {exc}") from exc


def _load_feature_columns(path: Path) -> list[str]:
    """Load the ordered feature column list saved by `preprocess.py`."""
    if not path.exists():
        raise RecommendationEngineError(f"Missing feature columns file at '{path}'. Run preprocess.py first.")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecommendationEngineError(f"Failed to read feature columns from '{path}': {exc}") from exc


def _load_label_encoder(path: Path) -> LabelEncoder:
    """Load the fitted `LabelEncoder` saved by `preprocess.py`."""
    if not path.exists():
        raise RecommendationEngineError(f"Missing label encoder at '{path}'. Run preprocess.py first.")
    try:
        return joblib.load(path)
    except (OSError, EOFError) as exc:
        raise RecommendationEngineError(f"Failed to load label encoder from '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Circuit loading and feature construction (mirrors inference.py)
# ---------------------------------------------------------------------------


def _load_circuit(qasm_path: Path) -> Any:
    """Parse the input `.qasm` file via `parser.load_qasm_file`.

    Raises:
        RecommendationEngineError: If the file is missing, not a
            `.qasm` file, or fails to parse.
    """
    try:
        return load_qasm_file(qasm_path)
    except FileNotFoundError as exc:
        raise RecommendationEngineError(f"QASM file not found: {qasm_path}") from exc
    except (ValueError, QasmParsingError) as exc:
        raise RecommendationEngineError(f"Invalid QASM file '{qasm_path}': {exc}") from exc


def _build_raw_feature_dict(circuit: Any) -> tuple[dict[str, float], dict[str, int], dict[str, Any]]:
    """Compute every feature this module CAN derive, via analyzer + feature_extractor.

    Returns:
        A tuple of (structural/ML feature values keyed by the training
        dataset's own column names, the circuit's raw gate-count dict,
        the full `analyzer.analyze_circuit` result -- reused later for
        gate-substitution rule checks that need raw gate counts by
        name, e.g. `gate_counts.get("ccx", 0)`).
    """
    analysis = analyze_circuit(circuit)
    features = extract_features(circuit)

    raw_features: dict[str, float] = {}
    for target_column, analysis_key in _STRUCTURAL_FEATURE_MAP.items():
        raw_features[target_column] = analysis[analysis_key]
    for target_column, feature_key in _ML_FEATURE_MAP.items():
        raw_features[target_column] = features[feature_key]

    return raw_features, analysis["gate_counts"], analysis


def _build_feature_vector(
    raw_features: dict[str, float],
    gate_counts: dict[str, int],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the model's exact input row, in the trained column order.

    Identical logic to `inference._build_feature_vector`: structural/ML
    columns come from `raw_features`, `gate_<name>` columns from
    `gate_counts`, and noise-derived columns are filled with a
    documented placeholder and recorded as unavailable.

    Raises:
        RecommendationEngineError: If a required column can't be
            resolved from any of the above.
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
            raise RecommendationEngineError(
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
) -> tuple[str, float, int, dict[str, float]]:
    """Run the model on one feature row and decode its class prediction.

    Returns:
        A tuple of (predicted class name, confidence, predicted class's
        integer index, full per-class probability dict).

    Raises:
        RecommendationEngineError: If the model has no `predict_proba`.
    """
    if not hasattr(model, "predict_proba"):
        raise RecommendationEngineError(
            f"Model type '{type(model).__name__}' has no 'predict_proba' method; "
            "cannot compute a confidence score or SHAP-based recommendations."
        )

    probabilities = model.predict_proba(feature_row)[0]
    class_names = [str(label) for label in label_encoder.classes_]
    class_probabilities = {
        class_name: float(probability) for class_name, probability in zip(class_names, probabilities)
    }

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, predicted_index, class_probabilities


# ---------------------------------------------------------------------------
# Explanation context (reused from explainability.py -- no SHAP computed here)
# ---------------------------------------------------------------------------


def _get_explainability_summary(
    config: RecommendationConfig,
    explainability_summary: ExplainabilitySummary | None,
) -> ExplainabilitySummary:
    """Return the SHAP/LIME explanation context this run needs.

    If the caller already has an `ExplainabilitySummary` -- e.g. a
    pipeline that already ran `explainability.explain_model` once for
    this training run and wants every circuit's recommendations to
    reuse that exact same result -- it is used as-is and
    `explainability.py` is not invoked again. Otherwise
    `explainability.explain_model` is called exactly once, which is the
    ONLY place, in this module or `explainability.py`, that ever
    constructs a SHAP explainer. This module never builds its own
    `shap.TreeExplainer` or `shap.Explainer`.

    Raises:
        RecommendationEngineError: If `explainability.explain_model`
            fails (e.g. missing trained-model or test-split artifacts).
    """
    if explainability_summary is not None:
        return explainability_summary

    explain_config = ExplainConfig(
        models_directory=config.models_directory,
        splits_directory=config.splits_directory,
        feature_columns_path=config.feature_columns_path,
        label_encoder_path=config.label_encoder_path,
        plots_directory=config.explainability_plots_directory,
        report_path=config.explainability_report_path,
    )
    try:
        return explain_model(explain_config)
    except ExplainabilityError as exc:
        raise RecommendationEngineError(
            f"Failed to obtain SHAP/LIME explanations from "
            f"explainability.explain_model: {exc}"
        ) from exc


def _dataset_level_shap_magnitude(
    summary: ExplainabilitySummary, predicted_class: str
) -> dict[str, float]:
    """Turn one of explainability.py's own per-class SHAP rankings into
    a feature -> magnitude lookup for this circuit's recommendations.

    `explainability.py` already computed a per-class SHAP ranking (mean
    |SHAP value| per feature, per class, across a test-set sample) the
    correct way for this project's model type -- see that module's
    two-tier `TreeExplainer` / fallback `shap.Explainer` strategy. This
    function reuses that ranking directly rather than computing anything
    new.

    Uses `summary.per_class_feature_ranking_full` (and, as a fallback,
    `summary.global_feature_ranking_full`) -- the COMPLETE, untruncated
    rankings covering every feature column -- rather than the
    `top_n_features`-truncated `per_class_feature_ranking` /
    `global_feature_ranking` fields those same objects also expose. The
    truncated fields exist only for `explainability.py`'s own report and
    plots; looking up an arbitrary rule's feature there would silently
    read as 0.0 for any feature ranked below `top_n_features` (e.g.
    15th, out of ~37 total columns), which is indistinguishable from the
    model genuinely assigning it zero importance and would cause
    `_build_recommendations` to incorrectly suppress a rule whose
    feature simply didn't make an arbitrary display cutoff.

    If this circuit's own predicted class is already the low-reliability
    class, that class's own ranking is the most direct signal: "what
    the model relies on when it predicts LOW". Otherwise, the
    LOW-reliability class's ranking is used instead -- it is the
    model's clearest dataset-wide signal for "what generally drives an
    unreliable prediction", and pairing it with this circuit's own
    structural thresholds (checked separately, in
    `_RecommendationRule.threshold_check`) keeps recommendations
    grounded in features this specific circuit actually exhibits.

    Returns:
        A dict of feature name -> non-negative SHAP-ranking magnitude,
        covering every feature column. Falls back to
        `summary.global_feature_ranking_full` if the requested class has
        no ranking (e.g. it never appeared as a predicted class in the
        explained test sample).
    """
    ranking_class = (
        predicted_class
        if predicted_class == _LOW_RELIABILITY_CLASS_NAME
        else _LOW_RELIABILITY_CLASS_NAME
    )
    ranking = summary.per_class_feature_ranking_full.get(ranking_class)
    if not ranking:
        ranking = summary.global_feature_ranking_full
    return dict(ranking)


def _select_matching_lime_example(
    summary: ExplainabilitySummary, predicted_class: str
) -> LimeExample | None:
    """Return explainability.py's LIME example for this predicted class.

    `explainability.explain_model` already produced one representative
    `LimeExample` per reliability class (the test-set circuit with the
    highest predicted probability for that class). Reusing the one that
    matches this circuit's own predicted class gives a concrete,
    already-computed local explanation to cite alongside the SHAP
    ranking, without running LIME again for this circuit.

    Returns:
        The matching `LimeExample`, or `None` if `explainability.py`
        did not produce one for this class (e.g. the class was absent
        from the explained test sample).
    """
    for example in summary.lime_examples:
        if example.predicted_class == predicted_class:
            return example
    return None


# ---------------------------------------------------------------------------
# Recommendation rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RecommendationRule:
    """One structural pattern this engine knows how to flag and advise on.

    Attributes:
        category: Machine-readable tag (see `Recommendation.category`).
        title: Short plain-language summary.
        feature_names: Feature column(s) this rule inspects, both for
            the structural-threshold check and for summing SHAP impact.
        threshold_check: Callable taking the raw feature dict (analyzer
            + feature_extractor's merged output) and returning True if
            this circuit is structurally significant enough for this
            rule to be a candidate.
        detail_builder: Callable taking the raw feature dict and
            returning the full, circuit-specific recommendation text.
    """

    category: str
    title: str
    feature_names: tuple[str, ...]
    threshold_check: Callable[[dict[str, Any]], bool]
    detail_builder: Callable[[dict[str, Any]], str]


def _detail_reduce_cx(raw: dict[str, Any]) -> str:
    cx_count = raw.get("gate_counts", {}).get("cx", 0)
    return (
        f"This circuit uses {cx_count} CX (CNOT) gate(s), the dominant "
        "two-qubit entangling primitive and typically the noisiest gate "
        "on real hardware (two-qubit depolarizing error rates are "
        "usually an order of magnitude higher than single-qubit rates). "
        "Consider: (1) reordering commuting gates so adjacent CX pairs "
        "can cancel or be merged during transpilation, (2) replacing "
        "chains of CX gates used only to build a specific entangling "
        "pattern (e.g. a GHZ-style chain) with an equivalent circuit "
        "using fewer CX gates where the target hardware topology "
        "allows, and (3) letting the transpiler's own optimization "
        "level (e.g. Qiskit's `optimization_level=3`) attempt CX "
        "cancellation and commutation-based reduction before execution."
    )


def _detail_reduce_depth(raw: dict[str, Any]) -> str:
    depth = raw.get("depth", 0)
    return (
        f"This circuit has a depth of {depth} layers. Depth is a direct "
        "proxy for how long the qubits must stay coherent, so deeper "
        "circuits accumulate more decoherence and gate-error exposure "
        "even without adding entangling gates. Consider: (1) "
        "parallelizing independent single-qubit operations that are "
        "currently serialized across separate layers, (2) removing "
        "gates that mutually cancel (e.g. adjacent X-X, H-H) via a "
        "transpiler optimization pass, and (3) re-expressing rotation "
        "sequences (e.g. consecutive RZ-RX-RZ on the same qubit) as a "
        "single combined rotation where the underlying math allows it."
    )


def _detail_reduce_entangling(raw: dict[str, Any]) -> str:
    entangling_count = raw.get("entangling_gates", 0)
    return (
        f"This circuit uses {entangling_count} entangling (multi-qubit) "
        "gate(s) in total (CX, CZ, CP, SWAP, and CCX combined). Every "
        "entangling gate carries substantially higher error than a "
        "single-qubit gate and also increases the circuit's "
        "susceptibility to crosstalk on real hardware. Consider "
        "reviewing whether every entangling operation is structurally "
        "necessary for the circuit's intended algorithm, or whether "
        "some can be replaced by classical control (e.g. mid-circuit "
        "measurement plus classically-conditioned single-qubit "
        "correction) where the target backend supports it."
    )


def _detail_gate_substitution(raw: dict[str, Any]) -> str:
    gate_counts = raw.get("gate_counts", {})
    ccx_count = gate_counts.get("ccx", 0)
    swap_count = gate_counts.get("swap", 0)
    lines = []
    if ccx_count >= _MIN_THREE_QUBIT_GATES_FOR_FLAG:
        lines.append(
            f"This circuit uses {ccx_count} CCX (Toffoli) gate(s). Each "
            "CCX typically decomposes into 6 CX gates plus several "
            "single-qubit gates during transpilation, making it one of "
            "the most expensive common primitives. Where the circuit's "
            "algorithm tolerates it, consider a relative-phase Toffoli "
            "(a cheaper CCX variant valid when only the sign of the "
            "target's phase matters, not an exact computational-basis "
            "flip) or a decomposition tailored to the target hardware's "
            "native gate set."
        )
    if swap_count >= _MIN_SWAP_GATES_FOR_FLAG:
        lines.append(
            f"This circuit uses {swap_count} SWAP gate(s). A SWAP "
            "typically decomposes into 3 CX gates. If a SWAP exists "
            "only to route two qubits' logical roles for hardware "
            "connectivity (rather than being part of the algorithm "
            "itself), consider whether the target backend's coupling "
            "map allows relabeling the logical-to-physical qubit "
            "mapping instead of inserting SWAP gates, or whether the "
            "transpiler's own routing pass can find a lower-SWAP "
            "layout."
        )
    if not lines:
        lines.append(
            "No specific high-cost gate substitution was identified for "
            "this circuit's gate mix; this note is included because the "
            "model attributed a meaningful reliability impact to one or "
            "more of this circuit's structural (non-gate-specific) "
            "features."
        )
    return " ".join(lines)


def _detail_reduce_measurements(raw: dict[str, Any]) -> str:
    measurement_count = raw.get("measurement_gates", 0)
    return (
        f"This circuit performs {measurement_count} measurement(s). "
        "Measurement (readout) error is typically the single largest "
        "per-operation error source on current hardware -- often several "
        "times higher than gate error. Consider: (1) measuring only the "
        "qubits whose classical outcome is actually consumed downstream "
        "rather than the full register, (2) deferring measurement to the "
        "very end of the circuit rather than interleaving mid-circuit "
        "measurements unless the algorithm specifically requires "
        "mid-circuit feedback, and (3) applying measurement-error "
        "mitigation (e.g. a calibration matrix) as a post-processing "
        "step if all measurements are structurally required."
    )


_RECOMMENDATION_RULES: tuple[_RecommendationRule, ...] = (
    _RecommendationRule(
        category="cx_gates",
        title="Reduce CX (CNOT) gate count",
        feature_names=("two_qubit_gates", "entangling_gates"),
        threshold_check=lambda raw: raw.get("gate_counts", {}).get("cx", 0) >= _MIN_CX_GATES_FOR_FLAG,
        detail_builder=_detail_reduce_cx,
    ),
    _RecommendationRule(
        category="depth",
        title="Reduce circuit depth",
        feature_names=("depth",),
        threshold_check=lambda raw: raw.get("depth", 0) >= _MIN_DEPTH_FOR_FLAG,
        detail_builder=_detail_reduce_depth,
    ),
    _RecommendationRule(
        category="entangling_gates",
        title="Reduce entangling gate count",
        feature_names=("entangling_gates",),
        threshold_check=lambda raw: raw.get("entangling_gates", 0) >= _MIN_ENTANGLING_GATES_FOR_FLAG,
        detail_builder=_detail_reduce_entangling,
    ),
    _RecommendationRule(
        category="gate_substitution",
        title="Replace costly gate sequences",
        feature_names=("three_qubit_gates", "two_qubit_gates"),
        threshold_check=lambda raw: (
            raw.get("gate_counts", {}).get("ccx", 0) >= _MIN_THREE_QUBIT_GATES_FOR_FLAG
            or raw.get("gate_counts", {}).get("swap", 0) >= _MIN_SWAP_GATES_FOR_FLAG
        ),
        detail_builder=_detail_gate_substitution,
    ),
    _RecommendationRule(
        category="measurements",
        title="Reduce number of measurements",
        feature_names=("measurement_gates",),
        threshold_check=lambda raw: raw.get("measurement_gates", 0) >= _MIN_MEASUREMENTS_FOR_FLAG,
        detail_builder=_detail_reduce_measurements,
    ),
)


def _shap_ranking_impact(
    shap_ranking: dict[str, float],
    feature_names: tuple[str, ...],
) -> float:
    """Sum a rule's magnitude from explainability.py's own SHAP ranking.

    `shap_ranking` (built by `_dataset_level_shap_magnitude`) is already
    oriented toward the low-reliability class -- its values are
    non-negative mean-|SHAP| magnitudes, not per-instance signed
    contributions -- so "more positive" always means "the model
    attributes more importance to this feature group in driving
    unreliable predictions", with no sign-flipping needed here.

    Args:
        shap_ranking: Output of `_dataset_level_shap_magnitude`.
        feature_names: The rule's feature column(s) to sum over.

    Returns:
        A float where more positive means "more evidence, per
        explainability.py's own SHAP ranking, that this feature group
        drives unreliable predictions". Rules are ranked by this value,
        descending.
    """
    return sum(shap_ranking.get(name, 0.0) for name in feature_names)


def _build_recommendations(
    raw_features_by_column: dict[str, Any],
    gate_counts: dict[str, int],
    shap_ranking: dict[str, float],
    max_recommendations: int,
) -> list[Recommendation]:
    """Evaluate every rule and return the ranked, thresholded recommendations.

    A rule only produces a `Recommendation` if both:
        1. `threshold_check` passes against the raw structural features
           (the circuit is structurally significant enough to mention).
        2. Its SHAP ranking magnitude (see `_shap_ranking_impact`) clears
           `_MIN_SHAP_MAGNITUDE` -- i.e. explainability.py's own
           attribution agrees this feature group is meaningfully
           associated with low-reliability predictions, not just
           structurally present in this circuit.

    Args:
        raw_features_by_column: The merged structural/ML feature dict,
            plus `gate_counts` folded in under that same key for rule
            detail-builders that need raw per-gate-name counts.
        gate_counts: The circuit's raw gate-name -> count dict (folded
            into `raw_features_by_column["gate_counts"]` for
            convenience, kept as its own parameter for clarity at the
            call site).
        shap_ranking: Output of `_dataset_level_shap_magnitude` -- a
            feature -> magnitude lookup reused from explainability.py,
            never recomputed here.
        max_recommendations: Cap on the number of recommendations
            returned, after ranking.

    Returns:
        Ranked (most impactful first) list of `Recommendation` objects,
        truncated to `max_recommendations`.
    """
    raw_with_gate_counts = dict(raw_features_by_column)
    raw_with_gate_counts["gate_counts"] = gate_counts

    candidates: list[Recommendation] = []
    for rule in _RECOMMENDATION_RULES:
        if not rule.threshold_check(raw_with_gate_counts):
            continue

        impact = _shap_ranking_impact(shap_ranking, rule.feature_names)
        if impact < _MIN_SHAP_MAGNITUDE:
            continue

        candidates.append(
            Recommendation(
                title=rule.title,
                detail=rule.detail_builder(raw_with_gate_counts),
                triggering_features=list(rule.feature_names),
                shap_impact=impact,
                category=rule.category,
            )
        )

    candidates.sort(key=lambda rec: rec.shap_impact, reverse=True)
    return candidates[:max_recommendations]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _resolve_circuit_display_name(circuit: Any, qasm_path: Path) -> str:
    """Return the circuit's explicit name, or the file stem if unnamed."""
    analysis = analyze_circuit(circuit)
    return analysis["name"] or qasm_path.stem


def _write_recommendation_report(path: Path, result: RecommendationResult) -> None:
    """Write the plain-text recommendation report.

    Raises:
        RecommendationEngineError: If the file cannot be written.
    """
    separator = "=" * 62
    lines: list[str] = [
        separator,
        "Quantum-Reliability-AI -- Recommendation Report",
        separator,
        "",
        f"Input Circuit      : {result.qasm_path}",
        f"Circuit Name       : {result.circuit_name}",
        f"Model Used         : {result.model_name}",
        f"Generated At (UTC) : {datetime.now(timezone.utc).isoformat()}",
        "",
        "-" * 62,
        "Prediction Basis",
        "-" * 62,
        f"Predicted Reliability Class : {result.predicted_class}",
        f"Confidence                  : {result.confidence:.4f} ({result.confidence * 100:.1f}%)",
        "",
        "-" * 62,
        "Explanation Basis (from explainability.py)",
        "-" * 62,
        f"SHAP Method Used     : {result.shap_method or 'unknown'}",
        f"Explanation Report   : {result.explanation_report_path or 'unavailable'}",
    ]

    if result.lime_example is not None:
        lines.append(
            f"Matching LIME Example: row {result.lime_example.row_index} "
            f"(true class: {result.lime_example.true_class}, "
            f"confidence: {result.lime_example.confidence:.4f})"
        )
        top_contributions = result.lime_example.contributions[:5]
        if top_contributions:
            lines.append("  Top local contributions for that representative circuit:")
            for feature_description, weight in top_contributions:
                lines.append(f"    - {feature_description}: {weight:+.5f}")
    else:
        lines.append("Matching LIME Example: none available for this predicted class.")

    lines += [
        "",
        "-" * 62,
        "Recommendations (ranked by explainability.py's SHAP ranking)",
        "-" * 62,
    ]

    if not result.recommendations:
        lines.append("")
        lines.append(
            "No structural changes are recommended: no feature group was "
            "both structurally significant and attributed a meaningful "
            "SHAP-ranking magnitude toward low-reliability predictions by "
            "explainability.py for this predicted class."
        )
    else:
        for rank, rec in enumerate(result.recommendations, start=1):
            lines += [
                "",
                f"{rank}. {rec.title}  [category: {rec.category}]",
                f"   SHAP-ranking magnitude (from explainability.py, toward low reliability): {rec.shap_impact:.5f}",
                f"   Triggering feature(s): {', '.join(rec.triggering_features)}",
                "",
            ]
            # Wrap the detail text to a readable width without any
            # external dependency, matching the plain, unadorned report
            # style the rest of the project's reports use.
            words = rec.detail.split()
            wrapped_line = "   "
            for word in words:
                if len(wrapped_line) + len(word) + 1 > 78:
                    lines.append(wrapped_line)
                    wrapped_line = "   " + word
                else:
                    wrapped_line = f"{wrapped_line} {word}".strip()
                    wrapped_line = "   " + wrapped_line.lstrip() if wrapped_line == word else wrapped_line
            if wrapped_line.strip():
                lines.append(wrapped_line)

    lines += [
        "",
        "-" * 62,
        "Notes and Limitations",
        "-" * 62,
        (
            "Recommendations are derived from this circuit's own structural "
            f"features combined with SHAP feature rankings and LIME local "
            f"explanations already computed by explainability.py against the "
            f"trained '{result.model_name}' classifier -- this module never "
            "computes SHAP itself. They are advisory, not automatically "
            "applied, and are not a guarantee that a given change will "
            "improve reliability on real hardware."
        ),
        (
            "The SHAP ranking used to rank recommendations is a dataset-wide "
            "(per-class) importance measure, not recomputed for this "
            "specific circuit; it is combined with this circuit's own "
            "structural thresholds so recommendations still reflect what "
            "this circuit actually exhibits. See the explanation report "
            "above for the full dataset-wide analysis."
        ),
        (
            "Gate-substitution suggestions are qualitative starting "
            "points; verifying that any substitution preserves the "
            "circuit's intended semantics is the user's responsibility."
        ),
    ]

    if result.unavailable_features:
        lines += [
            (
                "This module reuses only parser.py, analyzer.py, and "
                "feature_extractor.py -- it does not run noise_simulator.py "
                "on the input circuit. The following model input feature(s) "
                "could therefore not be computed and were filled with a "
                f"placeholder value ({_NOISE_PLACEHOLDER_VALUE}):"
            )
        ]
        for feature_name in result.unavailable_features:
            lines.append(f"  - {feature_name}")

    lines.append("")
    lines.append("Recommendation Complete.")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text("\n".join(lines) + "\n")
    except OSError as exc:
        raise RecommendationEngineError(
            f"Failed to write recommendation report to '{path}': {exc}"
        ) from exc


def _write_recommendation_json(path: Path, result: RecommendationResult) -> None:
    """Write a machine-readable companion to the text report.

    Mirrors `inference.py`'s convention of pairing a human-readable
    report with a structured JSON result.

    Raises:
        RecommendationEngineError: If the file cannot be written.
    """
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_engine_version": RECOMMENDATION_ENGINE_VERSION,
        "qasm_path": str(result.qasm_path),
        "circuit_name": result.circuit_name,
        "model_name": result.model_name,
        "predicted_class": result.predicted_class,
        "confidence": result.confidence,
        "explainability": {
            "shap_method": result.shap_method,
            "explanation_report_path": (
                str(result.explanation_report_path) if result.explanation_report_path else None
            ),
            "matching_lime_example": (
                {
                    "row_index": result.lime_example.row_index,
                    "true_class": result.lime_example.true_class,
                    "predicted_class": result.lime_example.predicted_class,
                    "confidence": result.lime_example.confidence,
                    "contributions": result.lime_example.contributions,
                }
                if result.lime_example is not None
                else None
            ),
        },
        "recommendations": [
            {
                "title": rec.title,
                "category": rec.category,
                "detail": rec.detail,
                "triggering_features": rec.triggering_features,
                "shap_impact": rec.shap_impact,
            }
            for rec in result.recommendations
        ],
        "unavailable_features": result.unavailable_features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        raise RecommendationEngineError(
            f"Failed to write recommendation JSON to '{path}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_recommendations(
    qasm_path: str | Path,
    config: RecommendationConfig | None = None,
    explainability_summary: ExplainabilitySummary | None = None,
) -> RecommendationResult:
    """Predict reliability and produce ranked improvement recommendations.

    End-to-end pipeline:
        1. Parse the `.qasm` file via `parser.load_qasm_file`.
        2. Run `analyzer.analyze_circuit` and
           `feature_extractor.extract_features`, and assemble the
           model's exact input feature row (identical construction to
           `inference.py`, including the documented noise-derived
           placeholder columns).
        3. Load the winning model and label encoder, and predict this
           circuit's reliability class and confidence (plain model
           inference -- no SHAP involved).
        4. Obtain a SHAP/LIME explanation context via
           `_get_explainability_summary` -- either the
           `explainability_summary` the caller supplied, or a fresh
           `explainability.explain_model` run. No SHAP is computed in
           this module.
        5. Evaluate the rule set (CX gates, depth, entangling gates,
           gate substitution, measurements), keeping only rules that are
           both structurally significant for this circuit and
           attributed a meaningful magnitude by explainability.py's own
           per-class SHAP ranking, ranked by that magnitude.
        6. Write `recommendation_report.txt` (and a companion
           `recommendation_result.json`) under `config.output_directory`.

    Args:
        qasm_path: Path to the `.qasm` file to analyze.
        config: Recommendation configuration. Defaults to
            `RecommendationConfig()` (reads from `models/` and
            `models/preprocessing/`, writes to `outputs/`).
        explainability_summary: An already-computed
            `explainability.ExplainabilitySummary` to reuse, e.g. from
            a caller that already ran `explainability.explain_model`
            earlier in the same session for the same trained model.
            When omitted, `explain_model` is called once here.

    Returns:
        A `RecommendationResult` with the prediction, ranked
        recommendations, and every output file's path.

    Raises:
        RecommendationEngineError: If the QASM file cannot be parsed,
            required trained-model artifacts are missing, the
            explainability context cannot be obtained, or any output
            file cannot be written.
    """
    active_config = config or RecommendationConfig()
    resolved_qasm_path = Path(qasm_path)
    models_directory = Path(active_config.models_directory)
    output_directory = Path(active_config.output_directory)

    circuit = _load_circuit(resolved_qasm_path)
    circuit_name = _resolve_circuit_display_name(circuit, resolved_qasm_path)

    raw_features, gate_counts, _analysis = _build_raw_feature_dict(circuit)
    feature_columns = _load_feature_columns(Path(active_config.feature_columns_path))
    feature_row, unavailable_features = _build_feature_vector(
        raw_features, gate_counts, feature_columns
    )

    model_name = _load_winning_model_name(models_directory / _MODEL_METRICS_FILENAME)
    model = _load_model(models_directory, model_name)
    label_encoder = _load_label_encoder(Path(active_config.label_encoder_path))

    predicted_class, confidence, _predicted_index, _class_probabilities = _predict(
        model, feature_row, label_encoder
    )

    summary = _get_explainability_summary(active_config, explainability_summary)
    shap_ranking = _dataset_level_shap_magnitude(summary, predicted_class)
    lime_example = _select_matching_lime_example(summary, predicted_class)

    recommendations = _build_recommendations(
        raw_features,
        gate_counts,
        shap_ranking,
        active_config.max_recommendations,
    )

    result = RecommendationResult(
        qasm_path=resolved_qasm_path,
        circuit_name=circuit_name,
        model_name=model_name,
        predicted_class=predicted_class,
        confidence=confidence,
        recommendations=recommendations,
        unavailable_features=unavailable_features,
        shap_method=summary.shap_method,
        explanation_report_path=summary.report_path,
        lime_example=lime_example,
    )

    report_path = output_directory / "recommendation_report.txt"
    result_json_path = output_directory / "recommendation_result.json"
    _write_recommendation_report(report_path, result)
    _write_recommendation_json(result_json_path, result)

    result.report_path = report_path
    result.result_json_path = result_json_path
    return result


def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the `__main__` CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ranked reliability-improvement recommendations "
        "for a single OpenQASM 3 file."
    )
    parser.add_argument("qasm_path", help="Path to the .qasm file to analyze.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Usage:
        python -m src.recommendation_engine path/to/circuit.qasm
    """
    args = _parse_cli_args(argv if argv is not None else sys.argv[1:])

    try:
        result = generate_recommendations(args.qasm_path)
    except RecommendationEngineError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("Recommendation Generation Complete")
    print("=" * 60)
    print(f"Circuit                  : {result.circuit_name}")
    print(f"Model Used               : {result.model_name}")
    print(f"Predicted Class          : {result.predicted_class}")
    print(f"Confidence               : {result.confidence:.4f} ({result.confidence * 100:.1f}%)")
    print(f"SHAP Method (reused)     : {result.shap_method}")
    print()
    if result.recommendations:
        print(f"Top recommendation(s), {len(result.recommendations)} total:")
        for rank, rec in enumerate(result.recommendations, start=1):
            print(f"  {rank}. {rec.title} (impact={rec.shap_impact:.5f})")
    else:
        print("No recommendations triggered for this circuit.")
    print()
    print(f"Report              : {result.report_path}")
    print(f"JSON                : {result.result_json_path}")
    print(f"Explanation Report  : {result.explanation_report_path}")


if __name__ == "__main__":
    main()
