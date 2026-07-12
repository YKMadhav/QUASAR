"""
explainability.py
-------------------
Single Responsibility:
    Load the winning trained model produced by `train_model.py`, together
    with `preprocess.py`'s saved test split, feature columns, and label
    encoder, and produce both global and local explanations of its
    predictions: SHAP-based feature-importance rankings and summary
    plots, LIME-based local explanations for a handful of representative
    circuits, a single ranked "what mattered most" feature table, a set
    of publication-quality figures, and one human-readable
    `explanation_report.txt` tying it all together.

This module intentionally does NOT:
    - Preprocess, split, or encode the dataset (see `preprocess.py`)
    - Train, retrain, or tune any model (see `train_model.py`) -- it only
      ever calls `.predict` / `.predict_proba` on the already-fitted
      winning model
    - Evaluate raw predictive performance -- accuracy, F1, ROC/AUC,
      confusion matrices, and the learning curve are `evaluate_model.py`'s
      job; this module only explains *why* the already-evaluated model
      predicts what it predicts, not *how well* it predicts
    - Modify any artifact produced by `preprocess.py`, `train_model.py`,
      or `evaluate_model.py`, or retrain / refit anything
    - Print anything to the console outside of its own progress
      reporting and `if __name__ == "__main__":` summary -- the same
      documented exception every long-running module in this project
      makes (see `dataset_generator.py`, `circuit_generator.py`,
      `train_model.py`, `evaluate_model.py`)

Where this fits in the pipeline:
    Circuit Generator -> Parser -> Analyzer -> Feature Extractor
    -> Noise Simulator -> Dataset Generator -> Preprocessing
    -> Model Training -> Model Evaluation -> Explainability (this module)
    -> Inference / Machine Learning consumers (future)

Why this module was rewritten (read before trusting "TreeExplainer" in
older versions of this file):
    The previous implementation assumed `shap.TreeExplainer` always
    supports multiclass tree ensembles, including
    `GradientBoostingClassifier`. That assumption is NOT safe in
    general: scikit-learn's `GradientBoostingClassifier` fits one
    regression tree per class per boosting round, and depending on the
    installed `shap` version, `TreeExplainer` can raise outright for
    this "multiclass, additive, one-vs-rest" structure, return an
    output shape that does not cleanly reconcile with the number of
    classes, or (rarely) silently mis-attribute values across classes.
    This rewrite never assumes `TreeExplainer` will work for any given
    model -- it always tries it, and always verifies the shape of what
    comes back before trusting it. Any incompatibility -- an outright
    exception, an unrecognized output shape, or a shape that doesn't
    match the number of classes -- triggers an automatic, silent (to
    the caller; logged in the report) fallback to `shap.Explainer`
    wrapped around the model's own `.predict_proba`, which works for
    literally any classifier that exposes `predict_proba`, tree-based
    or not. This module is therefore correct for the current winning
    model (`GradientBoostingClassifier`, 3-class) and will keep working
    unchanged if a future training run picks a different winner.

Design summary:
    - The winning model's identity and estimator are never hardcoded:
      both are read from `model_metrics.pkl` / `<name>.pkl`, exactly the
      way `evaluate_model.py` and `inference.py` do it.
    - SHAP computation is a two-tier strategy, see
      `_compute_shap_values`:
        1. Attempt `shap.TreeExplainer(model)`. If it raises, or if its
           output shape cannot be reconciled with the model's known
           class count, this attempt is abandoned (never propagated to
           the caller as a crash).
        2. Automatically fall back to `shap.Explainer(model.predict_proba,
           masker)` -- a model-agnostic explainer (Permutation/Kernel
           depending on the installed SHAP version) that only requires
           `predict_proba`, so it is correct for ANY scikit-learn-style
           classifier regardless of internal structure. A small,
           reproducible background sample is used as the masker so this
           stays tractable even though it does not exploit tree
           structure the way `TreeExplainer` does.
      Whichever tier actually produced the values is recorded (as
      `shap_method`) and surfaced in the report, so the report is
      always honest about how the numbers were computed.
    - Every SHAP output -- regardless of which tier produced it, and
      regardless of whether the installed SHAP version returns a
      Python list of per-class arrays, a single 3-D array, or a
      `shap.Explanation` object -- is normalized in one place
      (`_normalize_shap_output`) into one consistent representation: a
      dict of class name -> array of shape (n_samples, n_features).
      Every downstream ranking, plot, and report section consumes only
      this normalized representation and never touches SHAP's raw
      output shape again.
    - LIME: `lime.lime_tabular.LimeTabularExplainer` is used to explain
      a small number of individual circuits chosen to be representative
      -- by default, the test-set row with the highest predicted
      probability for each reliability class -- so the report shows one
      concrete, readable local explanation per class. LIME only ever
      calls `model.predict_proba`, so it is unaffected by whichever SHAP
      tier was used and needs no model-type handling of its own.
    - "Which features contributed most to the prediction" is answered at
      two different, complementary scopes, and both are surfaced in the
      report and the figures: globally via mean absolute SHAP value per
      feature (a dataset-wide ranking, per class and combined), and
      locally via each LIME explanation's own per-instance ranked,
      signed contributions for one specific circuit's prediction.
    - Every plot uses the same "publication style" `evaluate_model.py`
      establishes (headless Agg backend, consistent DPI/fonts/palette),
      applied once via `_apply_publication_style`, so explainability
      figures visually match the evaluation figures already produced
      for this project.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")  # Headless rendering: this module never opens a GUI window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

import shap
from lime.lime_tabular import LimeTabularExplainer
from tqdm.auto import tqdm as _tqdm_base

# Bumped whenever this module's explanation methodology or output schema
# changes in a way future consumers should know about. Mirrors
# `evaluate_model.EVALUATE_MODEL_VERSION` / `train_model.TRAIN_MODEL_VERSION`.
# 2.0.0: the SHAP computation strategy changed from "assume TreeExplainer
# always works" to "try TreeExplainer, verify, and automatically fall
# back to a model-agnostic Explainer".
# 2.1.0: `TreeExplainer` is now retried with `check_additivity=False`
# before ever falling back. In practice, `GradientBoostingClassifier`
# was triggering the fallback not because `TreeExplainer` is
# incompatible with it, but because SHAP's internal additivity
# sanity-check can fail on this model type due to floating-point drift
# in its raw margin output -- a false alarm, not a real incompatibility.
# Skipping that check keeps the exact same tree-based SHAP computation
# and avoids the much slower model-agnostic fallback in the common
# case. The fallback path itself also now suppresses noisy
# "X does not have valid feature names" sklearn warnings and accepts an
# explicit `max_evals` bound so its (inherently slower) runtime is
# predictable rather than open-ended.
# 2.2.0: `ExplainabilitySummary` now also carries the complete,
# untruncated global/per-class SHAP rankings (`global_feature_ranking_full`
# / `per_class_feature_ranking_full`), alongside the existing
# `top_n_features`-truncated ones used for the report and plots. A
# downstream consumer (recommendation_engine.py) was reading a specific
# feature's magnitude out of the truncated ranking, where any feature
# ranked below `top_n_features` (default 15 of ~37 columns) silently
# read as 0.0 -- indistinguishable from genuine zero importance. The
# full ranking is the correct source for that use case; the truncated
# fields' behavior and meaning are unchanged for report/plot purposes.
EXPLAINABILITY_VERSION = "2.2.0"

# Default locations, matching every upstream module's own output layout
# exactly -- this module reads those artifacts verbatim.
_DEFAULT_MODELS_DIRECTORY = Path("models")
_DEFAULT_PREPROCESSING_DIRECTORY = _DEFAULT_MODELS_DIRECTORY / "preprocessing"
_DEFAULT_SPLITS_DIRECTORY = _DEFAULT_PREPROCESSING_DIRECTORY / "splits"
_DEFAULT_PLOTS_DIRECTORY = Path("plots") / "explainability"
_DEFAULT_REPORT_PATH = Path("reports") / "explanation_report.txt"

_MODEL_METRICS_FILENAME = "model_metrics.pkl"
_CLASSIFICATION_TARGET_COLUMN = "reliability_class"


class _ShapProgressTqdm(_tqdm_base):
    """Custom tqdm subclass that redirects SHAP's progress to a callback.

    SHAP's PermutationExplainer uses tqdm internally to show progress.
    This wrapper captures those updates and forwards them to a callback
    function so the Streamlit dashboard can display a live progress bar.
    """

    _progress_callback: Callable[[str, float], None] | None = None
    _desc: str = ""

    def __init__(self, *args, **kwargs):
        self._progress_callback = kwargs.pop("shap_progress_callback", None)
        super().__init__(*args, **kwargs)

    def set_description(self, desc=None, refresh=True):
        self._desc = desc or self._desc
        super().set_description(desc, refresh)

    def update(self, n=1):
        super().update(n)
        if self._progress_callback and self.total:
            pct = self.n / self.total
            self._progress_callback(self._desc, pct)

    def close(self):
        if self._progress_callback and self.total:
            self._progress_callback(self._desc, 1.0)
        super().close()

# Names of "TreeExplainer" identifying which SHAP tier actually produced
# a given set of values -- recorded verbatim in `ExplainabilitySummary`
# and the written report, so the report is always explicit and honest
# about how the numbers were computed rather than silently assuming.
_METHOD_TREE_EXPLAINER = "shap.TreeExplainer"
_METHOD_FALLBACK_EXPLAINER = "shap.Explainer(predict_proba) [model-agnostic fallback]"

# Publication-style figure defaults, matching `evaluate_model.py`'s own
# constants exactly, so evaluation and explainability figures form one
# visually coherent report set.
_FIGURE_DPI = 300
_FIGURE_FACECOLOR = "white"
_PALETTE = ("#2E5EAA", "#D65F5F", "#3CA070", "#E1A340", "#7B6FD1")


class ExplainabilityError(Exception):
    """Raised when explanations cannot be produced as configured.

    Kept as a project-specific exception -- mirroring `EvaluationError`
    (evaluate_model.py), `TrainModelError` (train_model.py), and
    `InferenceError` (inference.py) -- so callers can catch one stable
    error type regardless of which internal step failed (missing
    artifacts, every SHAP strategy failing, a write failure). Note that
    an individual SHAP strategy being incompatible with the model is NOT
    one of those failure modes -- see `_compute_shap_values`, which
    automatically falls back rather than raising for that case.
    """


@dataclass(frozen=True)
class ExplainConfig:
    """Configuration for one explainability run.

    Attributes:
        models_directory: Directory containing the winning model
            (`<name>.pkl`) and `model_metrics.pkl`, as written by
            `train_model.py`.
        splits_directory: Directory containing `X_test.csv` / `y_test.csv`,
            as written by `preprocess.py`. The training split is not
            needed here -- explanations are computed against held-out
            data, consistent with `evaluate_model.py`'s own convention
            of reporting against the test set.
        feature_columns_path: Path to the feature column list JSON
            written by `preprocess.py`.
        label_encoder_path: Path to the fitted `LabelEncoder` `.joblib`
            file written by `preprocess.py`.
        plots_directory: Directory to save every explainability plot
            into. Created if it doesn't exist.
        report_path: Path to write the plain-text explanation report to.
            Its parent directory is created if it doesn't exist.
        shap_sample_size: Maximum number of test rows used to compute
            SHAP values. Sampled uniformly at random (seeded by
            `random_seed`) if the test split is larger than this.
        shap_background_size: Maximum number of test rows used as the
            "background" / masker reference distribution if the
            fallback model-agnostic `shap.Explainer` path is used.
            Ignored entirely if `TreeExplainer` succeeds (it needs no
            background set). Kept small since the fallback path's cost
            scales with background size.
        shap_fallback_max_evals: Upper bound on model evaluations per
            row for the fallback model-agnostic `shap.Explainer` path
            only (ignored entirely if `TreeExplainer` succeeds, which is
            the common case for this project's tree-ensemble
            candidates). `"auto"` lets SHAP choose its own default
            (`2 * num_features + 1`); pass a smaller `int` for a faster,
            lower-precision run if the fallback path is ever triggered
            and its default runtime is too slow.
        top_n_features: Number of top features to display on the global
            SHAP importance plots and in the report's ranking tables.
        lime_num_features: Number of features LIME includes in each
            local explanation (both the plot and the report).
        random_seed: Seed used for the SHAP sample draw and for LIME's
            own internal perturbation sampling, for reproducibility.
    """

    models_directory: str | Path = _DEFAULT_MODELS_DIRECTORY
    splits_directory: str | Path = _DEFAULT_SPLITS_DIRECTORY
    feature_columns_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "feature_columns.json"
    label_encoder_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "label_encoder.joblib"
    plots_directory: str | Path = _DEFAULT_PLOTS_DIRECTORY
    report_path: str | Path = _DEFAULT_REPORT_PATH
    shap_sample_size: int = 500
    shap_background_size: int = 100
    shap_fallback_max_evals: int | str = "auto"
    top_n_features: int = 15
    lime_num_features: int = 10
    random_seed: int | None = 42


@dataclass
class LimeExample:
    """One LIME local explanation, for one representative test-set circuit.

    Attributes:
        row_index: Positional index of this circuit within the test
            split (`X_test.csv`), for traceability.
        true_class: The circuit's actual `reliability_class` label.
        predicted_class: The model's predicted label for this circuit.
        confidence: The model's predicted probability for
            `predicted_class`.
        contributions: Ordered list of (feature, weight) pairs from
            LIME's local linear surrogate, sorted by descending absolute
            weight. A positive weight means that feature value pushed
            the prediction toward `predicted_class`; negative means it
            pushed away from it.
        plot_path: Path to this example's saved LIME explanation figure.
    """

    row_index: int
    true_class: str
    predicted_class: str
    confidence: float
    contributions: list[tuple[str, float]] = field(default_factory=list)
    plot_path: Path | None = None


@dataclass
class ExplainabilitySummary:
    """Summary of a completed explainability run.

    Attributes:
        model_name: Name of the explained (winning) model.
        model_type: The winning estimator's Python class name (e.g.
            "GradientBoostingClassifier"), recorded for auditability.
        shap_method: Which SHAP strategy actually produced the reported
            values -- `_METHOD_TREE_EXPLAINER` or
            `_METHOD_FALLBACK_EXPLAINER`. See module docstring.
        shap_fallback_reason: `None` if `TreeExplainer` succeeded
            outright; otherwise a short human-readable description of
            why it was abandoned (exception message or shape mismatch
            description), recorded so the report is honest about it.
        report_path: Path to the written explanation report.
        plot_paths: Mapping of plot identifier -> saved file path, for
            every figure this module produces.
        global_feature_ranking: The overall (all-class-averaged) SHAP
            feature ranking, as (feature name, mean |SHAP value|) pairs,
            sorted descending, truncated to `top_n_features`. Intended
            for display (the report and the top-N bar chart) -- do NOT
            use this to look up an arbitrary feature's magnitude, since
            a feature absent from this truncated list is not
            necessarily zero-importance, only outside the top N. Use
            `global_feature_ranking_full` for that instead.
        per_class_feature_ranking: The same display ranking computed
            separately per reliability class, also truncated to
            `top_n_features`. Same caveat as `global_feature_ranking`.
        global_feature_ranking_full: The complete (untruncated)
            all-class-averaged ranking, covering every feature column
            the model was trained on. This is the correct source for
            any downstream consumer (e.g. `recommendation_engine.py`)
            that needs a specific feature's actual SHAP magnitude --
            looking a feature up in the truncated ranking above would
            silently read as 0.0 for any feature ranked below
            `top_n_features`, which is not the same thing as the model
            genuinely assigning it zero importance.
        per_class_feature_ranking_full: The complete (untruncated)
            per-class ranking, same rationale as
            `global_feature_ranking_full`.
        lime_examples: One `LimeExample` per reliability class present
            in the test set.
    """

    model_name: str
    model_type: str
    shap_method: str
    shap_fallback_reason: str | None
    report_path: Path
    plot_paths: dict[str, Path] = field(default_factory=dict)
    global_feature_ranking: list[tuple[str, float]] = field(default_factory=list)
    per_class_feature_ranking: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    global_feature_ranking_full: list[tuple[str, float]] = field(default_factory=list)
    per_class_feature_ranking_full: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    lime_examples: list[LimeExample] = field(default_factory=list)


@dataclass(frozen=True)
class LocalShapContribution:
    """One signed SHAP contribution for a single explained circuit."""

    feature: str
    value: float
    shap_value: float


@dataclass(frozen=True)
class LocalShapExplanation:
    """Circuit-specific SHAP explanation for one model prediction."""

    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    shap_method: str
    shap_fallback_reason: str | None
    contributions: list[LocalShapContribution]
    unavailable_features: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading trained artifacts (mirrors evaluate_model.py / inference.py)
# ---------------------------------------------------------------------------


def _load_winning_model_name(model_metrics_path: Path) -> str:
    """Read which candidate won from `train_model.py`'s saved metrics file."""
    if not model_metrics_path.exists():
        raise ExplainabilityError(
            f"Missing model metrics at '{model_metrics_path}'. Run "
            "train_model.py before explainability.py."
        )
    try:
        payload = joblib.load(model_metrics_path)
    except (OSError, EOFError) as exc:
        raise ExplainabilityError(
            f"Failed to load model metrics from '{model_metrics_path}': {exc}"
        ) from exc

    winning_model = payload.get("winning_model")
    if not winning_model:
        raise ExplainabilityError(
            f"'{model_metrics_path}' does not contain a 'winning_model' key."
        )
    return str(winning_model)


def _load_model(models_directory: Path, model_name: str) -> ClassifierMixin:
    """Load the winning model's fitted estimator from `models/<name>.pkl`."""
    model_path = models_directory / f"{model_name}.pkl"
    if not model_path.exists():
        raise ExplainabilityError(
            f"Winning model file not found at '{model_path}'. Run "
            "train_model.py before explainability.py."
        )
    try:
        return joblib.load(model_path)
    except (OSError, EOFError) as exc:
        raise ExplainabilityError(f"Failed to load model from '{model_path}': {exc}") from exc


def _load_feature_columns(path: Path) -> list[str]:
    """Load the ordered feature column list saved by `preprocess.py`."""
    if not path.exists():
        raise ExplainabilityError(f"Missing feature columns file at '{path}'.")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExplainabilityError(f"Failed to read feature columns from '{path}': {exc}") from exc


def _load_label_encoder(path: Path) -> LabelEncoder:
    """Load the fitted `LabelEncoder` saved by `preprocess.py`."""
    if not path.exists():
        raise ExplainabilityError(f"Missing label encoder at '{path}'.")
    try:
        return joblib.load(path)
    except (OSError, EOFError) as exc:
        raise ExplainabilityError(f"Failed to load label encoder from '{path}': {exc}") from exc


def _load_csv(path: Path, description: str) -> pd.DataFrame:
    """Load one preprocessing split CSV, raising a clear error if absent."""
    if not path.exists():
        raise ExplainabilityError(
            f"Missing {description} at '{path}'. Run preprocess.py first."
        )
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ExplainabilityError(f"Failed to read {description} from '{path}': {exc}") from exc


@dataclass
class _ExplainData:
    """In-memory container for everything this module needs to explain the model."""

    x_test: pd.DataFrame
    y_test_encoded: np.ndarray
    y_test_labels: np.ndarray
    feature_columns: list[str]
    label_encoder: LabelEncoder
    class_names: list[str]


def _load_explain_data(config: ExplainConfig) -> _ExplainData:
    """Load every artifact needed to explain the winning model's test predictions.

    Raises:
        ExplainabilityError: If any required file is missing, unreadable,
            or inconsistent with the saved feature column list.
    """
    splits_dir = Path(config.splits_directory)

    x_test = _load_csv(splits_dir / "X_test.csv", "test feature matrix")
    y_test_df = _load_csv(splits_dir / "y_test.csv", "test targets")

    feature_columns = _load_feature_columns(Path(config.feature_columns_path))
    label_encoder = _load_label_encoder(Path(config.label_encoder_path))

    if list(x_test.columns) != feature_columns:
        raise ExplainabilityError(
            "Test feature matrix columns do not match feature_columns.json. "
            "Re-run preprocess.py to regenerate consistent artifacts."
        )

    if _CLASSIFICATION_TARGET_COLUMN not in y_test_df.columns:
        raise ExplainabilityError(
            f"'{_CLASSIFICATION_TARGET_COLUMN}' column missing from y_test.csv."
        )

    label_column = f"{_CLASSIFICATION_TARGET_COLUMN}_label"
    if label_column in y_test_df.columns:
        raw_labels = y_test_df[label_column].to_numpy()
    else:
        # Fall back to decoding the encoded column if the raw-label
        # convenience column preprocess.py normally writes isn't present.
        raw_labels = label_encoder.inverse_transform(
            y_test_df[_CLASSIFICATION_TARGET_COLUMN].to_numpy()
        )

    return _ExplainData(
        x_test=x_test.reset_index(drop=True),
        y_test_encoded=y_test_df[_CLASSIFICATION_TARGET_COLUMN].to_numpy(),
        y_test_labels=raw_labels,
        feature_columns=feature_columns,
        label_encoder=label_encoder,
        class_names=[str(label) for label in label_encoder.classes_],
    )


# ---------------------------------------------------------------------------
# Plot styling (matches evaluate_model.py exactly)
# ---------------------------------------------------------------------------


def _apply_publication_style() -> None:
    """Configure matplotlib rcParams for consistent, publication-quality figures.

    Identical to `evaluate_model._apply_publication_style` so every plot
    across the project's evaluation and explainability reports shares one
    visual language.
    """
    plt.rcParams.update(
        {
            "figure.dpi": _FIGURE_DPI,
            "savefig.dpi": _FIGURE_DPI,
            "figure.facecolor": _FIGURE_FACECOLOR,
            "axes.facecolor": _FIGURE_FACECOLOR,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.6,
            "font.size": 10,
            "font.family": "sans-serif",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure to disk with a tight layout, then close it.

    Raises:
        ExplainabilityError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=_FIGURE_DPI, facecolor=_FIGURE_FACECOLOR)
    except OSError as exc:
        raise ExplainabilityError(f"Failed to write plot to '{path}': {exc}") from exc
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# SHAP: sampling helpers
# ---------------------------------------------------------------------------


def _sample_test_rows(
    x_test: pd.DataFrame, sample_size: int, random_seed: int | None
) -> pd.DataFrame:
    """Draw a bounded, reproducible random sample of test rows for SHAP.

    Returns the full test set unchanged if it already has at most
    `sample_size` rows -- sampling is purely a scalability guard, not a
    requirement.
    """
    if len(x_test) <= sample_size:
        return x_test.reset_index(drop=True)
    return x_test.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SHAP: output normalization (shared by every strategy / SHAP version)
# ---------------------------------------------------------------------------


def _unwrap_explanation(raw: Any) -> Any:
    """Unwrap a `shap.Explanation` object (new-style SHAP API) to a raw array.

    Newer `shap.Explainer(...)` calls return an `Explanation` object
    rather than a bare array/list; older `TreeExplainer.shap_values(...)`
    calls return a bare array or list directly. Normalizing this one
    difference here means every downstream shape-handling branch in
    `_normalize_shap_output` only ever has to deal with arrays/lists.
    """
    if hasattr(raw, "values"):
        return raw.values
    return raw


def _normalize_shap_output(
    raw: Any, num_features: int, class_names: list[str]
) -> dict[str, np.ndarray]:
    """Normalize any SHAP output shape into {class_name: (n_samples, n_features)}.

    Handles every shape this project has observed across SHAP versions
    and explainer types:
        - A Python list of `num_classes` arrays, each
          (n_samples, n_features) -- classic `TreeExplainer.shap_values`
          multiclass output.
        - A single 3-D array of shape (n_samples, n_features, n_classes)
          -- newer `TreeExplainer` / `Explainer` multiclass output.
        - A single 3-D array of shape (n_classes, n_samples, n_features)
          -- an older alternative layout some SHAP versions used.
        - A single 2-D array of shape (n_samples, n_features), only
          valid when there are exactly 2 classes (binary: SHAP reports
          only the positive class, and the negative class is its
          negation).

    Args:
        raw: SHAP's raw output (already unwrapped via
            `_unwrap_explanation` if it came from a `shap.Explanation`).
        num_features: Expected number of feature columns.
        class_names: Class labels, in encoder order.

    Returns:
        A dict mapping each class name to its own (n_samples,
        n_features) SHAP value array.

    Raises:
        ValueError: If the shape cannot be reconciled with
            `class_names` / `num_features`. Callers are responsible for
            deciding whether that should trigger a fallback strategy or
            a hard failure -- this function itself never falls back.
    """
    num_classes = len(class_names)

    if isinstance(raw, list):
        if len(raw) != num_classes:
            raise ValueError(
                f"SHAP returned {len(raw)} per-class arrays, expected "
                f"{num_classes} (one per class)."
            )
        arrays = [np.asarray(v) for v in raw]
        for array in arrays:
            if array.ndim != 2 or array.shape[1] != num_features:
                raise ValueError(
                    f"One of SHAP's per-class arrays has shape {array.shape}, "
                    f"expected (n_samples, {num_features})."
                )
        return dict(zip(class_names, arrays))

    array = np.asarray(raw)

    if array.ndim == 3:
        if array.shape[-1] == num_classes and array.shape[1] == num_features:
            # (n_samples, n_features, n_classes)
            return {
                name: array[:, :, class_index]
                for class_index, name in enumerate(class_names)
            }
        if array.shape[0] == num_classes and array.shape[-1] == num_features:
            # (n_classes, n_samples, n_features)
            return {name: array[class_index] for class_index, name in enumerate(class_names)}
        raise ValueError(
            f"Could not interpret 3-D SHAP output of shape {array.shape} for "
            f"{num_classes} classes and {num_features} features."
        )

    if array.ndim == 2:
        if array.shape[1] != num_features:
            raise ValueError(
                f"2-D SHAP output has shape {array.shape}, expected "
                f"(n_samples, {num_features})."
            )
        if num_classes == 2:
            # Binary classification: some SHAP versions/models return a
            # single (n_samples, n_features) array for the positive
            # class only; the negative class is its exact negation.
            return {class_names[0]: -array, class_names[1]: array}
        raise ValueError(
            f"SHAP returned a single 2-D array of shape {array.shape}, but "
            f"there are {num_classes} classes -- a 2-D array is only "
            "unambiguous for binary classification."
        )

    raise ValueError(f"Unrecognized SHAP output with ndim={array.ndim}, shape={array.shape}.")


# ---------------------------------------------------------------------------
# SHAP: two-tier computation strategy (TreeExplainer, then automatic
# fallback to a model-agnostic Explainer)
# ---------------------------------------------------------------------------


def _describe_model_type(model: ClassifierMixin) -> str:
    """Return the estimator's class name, for logging and the report."""
    return type(model).__name__


def _attempt_tree_explainer(
    model: ClassifierMixin,
    sample: pd.DataFrame,
    class_names: list[str],
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """Attempt `shap.TreeExplainer` and validate its output shape.

    This is a pure "try it and verify" attempt: it never raises to the
    caller. Any failure -- an exception from SHAP itself, or an output
    shape that can't be reconciled with `class_names` -- is caught and
    reported back as `(None, reason)` so the caller can automatically
    move on to the fallback strategy instead.

    Args:
        model: The fitted winning classifier.
        sample: The (possibly down-sampled) test feature matrix.
        class_names: Class labels, in encoder order.

    Returns:
        A tuple of (normalized per-class SHAP dict, None) on success, or
        (None, short human-readable failure reason) if `TreeExplainer`
        is not usable for this model.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            try:
                # `check_additivity=True` (SHAP's default) verifies that
                # each row's SHAP values sum to `model output - expected
                # value`. For `GradientBoostingClassifier` in particular,
                # this check is a well-known source of false-positive
                # failures: the model's raw margin/log-odds output can
                # drift from the SHAP decomposition by a tiny
                # floating-point amount that has nothing to do with the
                # SHAP values themselves being wrong. Retrying with
                # `check_additivity=False` keeps the exact same
                # tree-based SHAP computation -- it only skips that
                # internal sanity check -- and is the standard, accepted
                # way to use TreeExplainer with this model type. Doing
                # this here (before ever falling back) avoids paying for
                # the much slower model-agnostic fallback for a "failure"
                # that was never about model incompatibility.
                raw_shap_values = explainer.shap_values(sample, check_additivity=False)
            except TypeError:
                # Older SHAP versions don't accept `check_additivity` as
                # a keyword on `shap_values` at all -- fall back to the
                # plain call for those, so this still works across SHAP
                # versions rather than hard-requiring a recent one.
                raw_shap_values = explainer.shap_values(sample)
    except Exception as exc:  # noqa: BLE001 -- any backend failure just means "try the fallback"
        return None, f"shap.TreeExplainer raised {type(exc).__name__}: {exc}"

    try:
        unwrapped = _unwrap_explanation(raw_shap_values)
        normalized = _normalize_shap_output(unwrapped, sample.shape[1], class_names)
    except ValueError as exc:
        return None, f"shap.TreeExplainer output shape was not usable: {exc}"

    return normalized, None


def _build_background_sample(
    x_test: pd.DataFrame, background_size: int, random_seed: int | None
) -> pd.DataFrame:
    """Build a small, reproducible background reference set for the fallback explainer.

    The model-agnostic fallback path needs a background distribution to
    define "what does a typical/absent feature value look like" -- this
    is analogous to `TreeExplainer`'s implicit use of the training data
    path statistics, but has to be supplied explicitly for a
    perturbation-based explainer. Kept deliberately small
    (`background_size`, default 100) since this path's runtime cost
    scales directly with it.
    """
    if len(x_test) <= background_size:
        return x_test.reset_index(drop=True)
    return x_test.sample(n=background_size, random_state=random_seed).reset_index(drop=True)


def _make_dataframe_preserving_predict_proba(
    model: ClassifierMixin, feature_columns: list[str]
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap `model.predict_proba` so masked perturbations keep feature names.

    `shap`'s maskers pass plain `numpy` arrays into the wrapped callable
    (they have no notion of the original column names), but this
    project's models are fit on `pandas` DataFrames with named columns.
    Calling `model.predict_proba(numpy_array)` directly still works, but
    scikit-learn emits a `UserWarning` on every single call ("X does not
    have valid feature names") -- and the fallback path can make
    thousands of such calls, flooding the console. Re-wrapping each
    batch back into a DataFrame with the correct column order/names
    before calling `predict_proba` produces identical predictions
    without the warning spam.
    """

    def _predict_proba(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(data), columns=feature_columns)
        return model.predict_proba(frame)

    return _predict_proba


def _attempt_fallback_explainer(
    model: ClassifierMixin,
    sample: pd.DataFrame,
    background: pd.DataFrame,
    class_names: list[str],
    max_evals: int | str = "auto",
    progress_callback: Callable[[str], None] | None = None,
    shap_progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, np.ndarray]:
    """Compute SHAP values with a model-agnostic `shap.Explainer`.

    Wraps `model.predict_proba` (via a DataFrame-preserving adapter, see
    `_make_dataframe_preserving_predict_proba`) rather than the
    estimator object itself, so this path works identically for ANY
    classifier that exposes `predict_proba` -- tree ensemble or not --
    and is therefore a safe universal fallback whenever `TreeExplainer`
    turns out to be genuinely incompatible with the winning model.

    This path is a perturbation-based approximation (typically SHAP's
    `PermutationExplainer` under the hood), so it is inherently much
    slower per row than `TreeExplainer` -- its cost scales with the
    number of features, not just the number of rows. `max_evals` bounds
    that cost explicitly rather than leaving it uncapped.

    Args:
        model: The fitted winning classifier.
        sample: The (possibly down-sampled) test feature matrix to
            explain.
        background: A small reference/background sample used as the
            masker distribution.
        class_names: Class labels, in encoder order.
        max_evals: Upper bound on the number of model evaluations per
            explained row, passed straight through to `shap.Explainer`.
            `"auto"` lets SHAP pick (`2 * num_features + 1`, its own
            default); an explicit `int` trades explanation precision for
            a hard, predictable runtime ceiling.
        progress_callback: Optional callable invoked with a short status
            string once, before the (potentially slow) computation
            starts, so a caller knows why this may take a while.
        shap_progress_callback: Optional callable invoked with
            (description, progress_fraction) tuples as SHAP's internal
            PermutationExplainer updates its tqdm progress bar. Used by
            the Streamlit dashboard to display a live progress bar.

    Returns:
        A normalized per-class SHAP dict, exactly like
        `_attempt_tree_explainer`'s success case.

    Raises:
        ExplainabilityError: If this fallback itself fails outright
            (e.g. the model has no `predict_proba`, or SHAP cannot
            reconcile the output shape either) -- at that point there is
            no further, safer strategy left to try.
    """
    if not hasattr(model, "predict_proba"):
        raise ExplainabilityError(
            f"Model type '{_describe_model_type(model)}' exposes no "
            "'predict_proba' method; neither shap.TreeExplainer nor the "
            "model-agnostic shap.Explainer fallback can compute SHAP "
            "values without class probabilities."
        )

    if progress_callback:
        evals_description = "SHAP's default budget" if max_evals == "auto" else f"max_evals={max_evals}"
        progress_callback(
            f"Model-agnostic fallback is perturbation-based and scales with "
            f"feature count ({sample.shape[1]} features here) -- this can take "
            f"a while ({evals_description}). Reduce `shap_sample_size` / "
            "`shap_fallback_max_evals` in ExplainConfig for a faster (less "
            "precise) run."
        )

    feature_columns = list(sample.columns)
    predict_proba = _make_dataframe_preserving_predict_proba(model, feature_columns)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            masker = shap.maskers.Independent(background.to_numpy())
            explainer = shap.Explainer(
                predict_proba,
                masker,
                feature_names=feature_columns,
            )
            # Temporarily replace tqdm with our custom wrapper to capture
            # SHAP's PermutationExplainer progress and redirect it to the
            # Streamlit dashboard's live progress bar.
            import tqdm as _tqdm_module
            _original_tqdm = _tqdm_module.tqdm
            if shap_progress_callback:
                _tqdm_module.tqdm = lambda *args, **kwargs: _ShapProgressTqdm(
                    *args, **kwargs, shap_progress_callback=shap_progress_callback
                )
            try:
                raw_shap_values = explainer(sample.to_numpy(), max_evals=max_evals)
            finally:
                _tqdm_module.tqdm = _original_tqdm
    except Exception as exc:  # noqa: BLE001 -- this is the last strategy; report clearly
        raise ExplainabilityError(
            "The model-agnostic shap.Explainer fallback also failed to "
            f"compute SHAP values for model type "
            f"'{_describe_model_type(model)}': {type(exc).__name__}: {exc}"
        ) from exc

    try:
        unwrapped = _unwrap_explanation(raw_shap_values)
        return _normalize_shap_output(unwrapped, sample.shape[1], class_names)
    except ValueError as exc:
        raise ExplainabilityError(
            "The model-agnostic shap.Explainer fallback produced an "
            f"output shape that could not be interpreted: {exc}"
        ) from exc


def _compute_shap_values(
    model: ClassifierMixin,
    sample: pd.DataFrame,
    background: pd.DataFrame,
    class_names: list[str],
    fallback_max_evals: int | str = "auto",
    progress_callback: Callable[[str], None] | None = None,
    shap_progress_callback: Callable[[str, float], None] | None = None,
) -> tuple[dict[str, np.ndarray], str, str | None]:
    """Compute per-class SHAP values, automatically choosing a working strategy.

    Strategy (see module docstring for the full rationale):
        1. Try `shap.TreeExplainer(model)`, exact and fast for tree
           ensembles. Its output shape is always verified before being
           trusted.
        2. If step 1 raised or produced an unusable shape for ANY
           reason (this is exactly the failure mode the previous
           version of this module did not handle for multiclass
           `GradientBoostingClassifier`), automatically fall back to a
           model-agnostic `shap.Explainer` wrapped around
           `model.predict_proba`, which makes no assumption about the
           model's internal structure at all.

    This function never crashes because of a SHAP/model incompatibility
    -- an incompatibility at step 1 simply selects step 2. It only
    raises if step 2 *also* fails, which indicates a genuine, otherwise
    unrecoverable problem (e.g. no `predict_proba` at all).

    Args:
        model: The fitted winning classifier.
        sample: The (possibly down-sampled) test feature matrix to
            explain.
        background: Background/reference sample for the fallback path
            (unused, and not computed by the caller, if the
            `TreeExplainer` path succeeds).
        class_names: Class labels, in encoder order.
        fallback_max_evals: Passed straight through to the fallback
            path's `shap.Explainer` call, if it's reached -- see
            `ExplainConfig.shap_fallback_max_evals`.
        progress_callback: Optional callable invoked with short status
            strings (e.g. `print`) as each strategy is attempted.
        shap_progress_callback: Optional callable invoked with
            (description, progress_fraction) tuples for live progress
            updates from SHAP's PermutationExplainer.

    Returns:
        A tuple of:
            - The normalized per-class SHAP dict.
            - Which method actually produced it (`_METHOD_TREE_EXPLAINER`
              or `_METHOD_FALLBACK_EXPLAINER`).
            - `None` if `TreeExplainer` succeeded outright, otherwise the
              short human-readable reason it was abandoned.

    Raises:
        ExplainabilityError: If both strategies fail (see
            `_attempt_fallback_explainer`).
    """
    if progress_callback:
        progress_callback(
            f"Attempting shap.TreeExplainer for model type "
            f"'{_describe_model_type(model)}'..."
        )

    tree_result, tree_failure_reason = _attempt_tree_explainer(model, sample, class_names)

    if tree_result is not None:
        if progress_callback:
            progress_callback("shap.TreeExplainer succeeded; using exact tree-based SHAP values.")
        return tree_result, _METHOD_TREE_EXPLAINER, None

    if progress_callback:
        progress_callback(
            f"shap.TreeExplainer was not usable for this model "
            f"({tree_failure_reason}); falling back to a model-agnostic "
            "shap.Explainer(predict_proba)..."
        )

    fallback_result = _attempt_fallback_explainer(
        model, sample, background, class_names, fallback_max_evals, progress_callback,
        shap_progress_callback=shap_progress_callback,
    )

    if progress_callback:
        progress_callback("Model-agnostic shap.Explainer fallback succeeded.")

    return fallback_result, _METHOD_FALLBACK_EXPLAINER, tree_failure_reason


# ---------------------------------------------------------------------------
# SHAP: feature ranking
# ---------------------------------------------------------------------------


def _rank_features_by_mean_abs_shap(
    shap_values: np.ndarray, feature_columns: list[str]
) -> list[tuple[str, float]]:
    """Rank features by mean absolute SHAP value, descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_columns, mean_abs.tolist()), key=lambda item: item[1], reverse=True)
    return ranking


def _combine_class_rankings(
    per_class_shap: dict[str, np.ndarray], feature_columns: list[str]
) -> list[tuple[str, float]]:
    """Build one overall feature ranking, averaged across every class.

    Averaging (rather than summing) each feature's mean-|SHAP| across
    classes keeps the overall score on the same scale as any individual
    class's own ranking, making the two directly comparable in the
    report.
    """
    stacked = np.stack(
        [np.abs(values).mean(axis=0) for values in per_class_shap.values()], axis=0
    )
    overall_scores = stacked.mean(axis=0)
    return sorted(
        zip(feature_columns, overall_scores.tolist()), key=lambda item: item[1], reverse=True
    )


# ---------------------------------------------------------------------------
# SHAP: plots
# ---------------------------------------------------------------------------


def _plot_global_shap_bar(
    overall_ranking: list[tuple[str, float]], top_n: int, path: Path
) -> None:
    """Render a horizontal bar chart of the top-N globally important features."""
    top_features = overall_ranking[:top_n][::-1]  # reverse so #1 plots at the top
    names = [name for name, _ in top_features]
    scores = [score for _, score in top_features]

    fig_height = max(4.0, 0.35 * len(top_features))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.barh(names, scores, color=_PALETTE[0])
    ax.set_xlabel("Mean |SHAP value| (average across classes)")
    ax.set_title(f"Top {len(top_features)} Features by Global SHAP Importance")

    _save_figure(fig, path)


def _plot_shap_importance_by_class(
    per_class_ranking: dict[str, list[tuple[str, float]]],
    feature_order: list[str],
    path: Path,
) -> None:
    """Render a grouped horizontal bar chart comparing per-class SHAP importance.

    `feature_order` (typically the overall top-N ranking) fixes which
    features appear and in what order, so every class's bars line up
    against the same feature axis for direct visual comparison.
    """
    class_names = list(per_class_ranking.keys())
    feature_order_top_first = feature_order[::-1]

    lookup = {
        class_name: dict(per_class_ranking[class_name]) for class_name in class_names
    }

    y_positions = np.arange(len(feature_order_top_first))
    bar_height = 0.8 / max(len(class_names), 1)

    fig_height = max(4.0, 0.4 * len(feature_order_top_first))
    fig, ax = plt.subplots(figsize=(7.5, fig_height))

    for class_index, class_name in enumerate(class_names):
        scores = [lookup[class_name].get(feature, 0.0) for feature in feature_order_top_first]
        offset = (class_index - (len(class_names) - 1) / 2) * bar_height
        ax.barh(
            y_positions + offset,
            scores,
            height=bar_height,
            color=_PALETTE[class_index % len(_PALETTE)],
            label=class_name,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_order_top_first)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Per-Class SHAP Feature Importance")
    ax.legend(loc="lower right")

    _save_figure(fig, path)


def _plot_shap_beeswarm(
    shap_values: np.ndarray,
    sample: pd.DataFrame,
    class_name: str,
    path: Path,
) -> None:
    """Render and save a SHAP beeswarm (summary) plot for one class.

    Delegates the actual plotting to `shap.summary_plot`, which already
    encodes feature-value-vs-impact color coding; this module only
    supplies the publication styling context and handles the save/close
    lifecycle so it's consistent with every other plot here. Works
    identically regardless of which of the two computation strategies
    produced `shap_values` -- by the time a plot function sees it, it is
    already a plain, normalized (n_samples, n_features) array.
    """
    fig = plt.figure(figsize=(7.5, max(4.0, 0.3 * sample.shape[1])))
    try:
        shap.summary_plot(
            shap_values,
            sample,
            feature_names=list(sample.columns),
            show=False,
            plot_size=None,
        )
        fig = plt.gcf()
        fig.suptitle(f"SHAP Beeswarm -- Predicted Class: {class_name}", y=1.02, fontweight="bold")
    except Exception as exc:  # noqa: BLE001 -- keep the run going even if one class's plot fails
        plt.close(fig)
        raise ExplainabilityError(
            f"Failed to render SHAP beeswarm plot for class '{class_name}': {exc}"
        ) from exc

    _save_figure(fig, path)


# ---------------------------------------------------------------------------
# LIME: representative instance selection and computation
# ---------------------------------------------------------------------------


def _select_representative_instances(
    model: ClassifierMixin, data: _ExplainData
) -> dict[str, int]:
    """Pick one representative test-set row per class for LIME.

    The representative for each class is the test row with the highest
    model-predicted probability for that class -- i.e. the circuit the
    model is *most confident* is LOW / MEDIUM / HIGH reliability -- so
    each LIME explanation shows a clear, prototypical case rather than a
    borderline or ambiguous one. This only calls `model.predict_proba`,
    so it is unaffected by which SHAP strategy was used above.

    Args:
        model: The fitted winning classifier.
        data: Loaded test data bundle.

    Returns:
        A dict mapping class name -> positional row index in `data.x_test`.
    """
    probabilities = model.predict_proba(data.x_test)
    representatives: dict[str, int] = {}
    for class_index, class_name in enumerate(data.class_names):
        best_row = int(np.argmax(probabilities[:, class_index]))
        representatives[class_name] = best_row
    return representatives


def _explain_instance_with_lime(
    explainer: LimeTabularExplainer,
    model: ClassifierMixin,
    row: pd.Series,
    predicted_class_index: int,
    num_features: int,
) -> tuple[list[tuple[str, float]], Any]:
    """Run LIME on one row and return its ranked, signed contributions.

    Returns:
        A tuple of (ranked (feature description, weight) pairs sorted by
        descending absolute weight, the raw LIME explanation object for
        plotting). Feature descriptions include LIME's own value-range
        condition (e.g. "depth > 42.00") so the report stays
        self-explanatory without cross-referencing the raw row.
    """
    explanation = explainer.explain_instance(
        row.to_numpy(),
        model.predict_proba,
        num_features=num_features,
        labels=(predicted_class_index,),
    )
    contributions = explanation.as_list(label=predicted_class_index)
    return sorted(contributions, key=lambda item: abs(item[1]), reverse=True), explanation


def _plot_lime_explanation(explanation: Any, class_name: str, path: Path) -> None:
    """Render and save LIME's own explanation figure for one instance.

    `explanation.as_pyplot_figure()` defaults to `label=1`, which assumes
    a binary classifier where label index 1 is always present. This
    explanation object was built with `explain_instance(..., labels=(predicted_class_index,))`
    (see `_explain_instance_with_lime`), so `explanation.local_exp` only
    ever contains that one label -- for a multiclass problem that label
    is frequently NOT 1, which is exactly what raised the `KeyError: 1`.
    Instead of assuming a label, we read whichever label(s) LIME actually
    populated in `local_exp` and render that one.
    """
    try:
        available_labels = list(explanation.local_exp.keys())
        if not available_labels:
            raise ValueError("LIME explanation has no labels in local_exp to plot.")
        label_to_plot = available_labels[0]

        fig = explanation.as_pyplot_figure(label=label_to_plot)
        fig.suptitle(f"LIME Explanation -- Predicted Class: {class_name}", fontweight="bold")
    except Exception as exc:  # noqa: BLE001
        raise ExplainabilityError(
            f"Failed to render LIME explanation plot for class '{class_name}': {exc}"
        ) from exc

    _save_figure(fig, path)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _format_ranking_lines(ranking: list[tuple[str, float]], indent: str = "  ") -> list[str]:
    """Format a (feature, score) ranking as aligned, numbered report lines."""
    lines = []
    for position, (feature, score) in enumerate(ranking, start=1):
        lines.append(f"{indent}{position:>2}. {feature:<30} {score:.5f}")
    return lines


def _format_lime_example_lines(example: LimeExample) -> list[str]:
    """Format one LIME example as a block of report lines."""
    lines = [
        f"Test Row Index      : {example.row_index}",
        f"True Class          : {example.true_class}",
        f"Predicted Class     : {example.predicted_class}",
        f"Confidence          : {example.confidence:.4f} ({example.confidence * 100:.1f}%)",
        "",
        "Top Local Contributions (feature condition -> signed weight):",
    ]
    for feature_condition, weight in example.contributions:
        direction = "supports" if weight > 0 else "opposes"
        lines.append(f"  {feature_condition:<45} {weight:+.5f}  ({direction} prediction)")
    if example.plot_path is not None:
        lines.append("")
        lines.append(f"Plot: {example.plot_path}")
    return lines


def _write_explanation_report(
    path: Path,
    model_name: str,
    model_type: str,
    shap_method: str,
    shap_fallback_reason: str | None,
    overall_ranking: list[tuple[str, float]],
    per_class_ranking: dict[str, list[tuple[str, float]]],
    lime_examples: list[LimeExample],
    shap_sample_rows: int,
    total_test_rows: int,
    plot_paths: dict[str, Path],
) -> None:
    """Write the plain-text explanation report summarizing every finding.

    Raises:
        ExplainabilityError: If the file cannot be written.
    """
    separator = "=" * 70
    lines: list[str] = [
        separator,
        "Quantum-Reliability-AI -- Model Explanation Report",
        separator,
        "",
        f"Explained Model         : {model_name}",
        f"Model Type              : {model_type}",
        f"Generated At (UTC)      : {datetime.now(timezone.utc).isoformat()}",
        f"Test Set Rows           : {total_test_rows}",
        f"SHAP Sample Rows        : {shap_sample_rows}",
        "",
        "-" * 70,
        "SHAP Computation Strategy",
        "-" * 70,
        f"Method Used             : {shap_method}",
    ]

    if shap_fallback_reason is None:
        lines.append(
            "shap.TreeExplainer computed exact SHAP values directly for "
            f"this '{model_type}' model; no fallback was necessary."
        )
    else:
        lines += [
            (
                "shap.TreeExplainer was attempted first (as it is exact and "
                "fast for tree ensembles) but was not usable for this "
                f"model, so this module automatically fell back to a "
                "model-agnostic shap.Explainer built around the model's "
                "own predict_proba -- no crash, no manual intervention "
                "required."
            ),
            f"Reason TreeExplainer was abandoned: {shap_fallback_reason}",
        ]

    lines += [
        "",
        "-" * 70,
        "Global Feature Importance (SHAP, averaged across classes)",
        "-" * 70,
        (
            "The features below moved the model's predictions the most, "
            "on average, across every circuit in the SHAP sample and "
            "every reliability class. This is the single best answer to "
            "'what does this model pay attention to, overall'."
        ),
        "",
    ]
    lines += _format_ranking_lines(overall_ranking)

    lines += [
        "",
        "-" * 70,
        "Per-Class Feature Importance (SHAP)",
        "-" * 70,
        (
            "The same ranking, computed separately for each reliability "
            "class -- useful when a feature matters a great deal for "
            "distinguishing one class (e.g. HIGH) but not another."
        ),
    ]
    for class_name, ranking in per_class_ranking.items():
        lines.append("")
        lines.append(f"Class: {class_name}")
        lines += _format_ranking_lines(ranking)

    lines += [
        "",
        "-" * 70,
        "Local Explanations (LIME) -- One Representative Circuit Per Class",
        "-" * 70,
        (
            "Each example below is the test-set circuit the model is "
            "most confident belongs to that reliability class. Local "
            "contributions come from LIME's own linear surrogate model "
            "fit around that single circuit -- they explain THIS "
            "prediction, not the model in general."
        ),
    ]
    for example in lime_examples:
        lines.append("")
        lines.append(f"--- {example.predicted_class} ---")
        lines += _format_lime_example_lines(example)

    lines += [
        "",
        "-" * 70,
        "Generated Plots",
        "-" * 70,
    ]
    for plot_name, plot_path in plot_paths.items():
        lines.append(f"{plot_name:<32}: {plot_path}")

    lines += [
        "",
        "-" * 70,
        "Notes and Limitations",
        "-" * 70,
        (
            f"SHAP values were computed with {shap_method}. See the "
            "'SHAP Computation Strategy' section above for whether the "
            "exact tree-based explainer or the model-agnostic fallback "
            "was used, and why."
        ),
        (
            f"SHAP values were computed on a random sample of "
            f"{shap_sample_rows} test rows (out of {total_test_rows}) for "
            "tractability; the global and per-class rankings above "
            "reflect that sample, not necessarily every row."
        ),
        (
            "LIME explanations are local: each one is only valid around "
            "the single circuit it explains, fit from perturbations of "
            "that circuit's own feature values, and should not be read "
            "as a global feature-importance statement."
        ),
        (
            "As with evaluate_model.py, this module never retrains or "
            "modifies the winning model -- it only calls .predict / "
            ".predict_proba on the estimator train_model.py already "
            "produced and saved."
        ),
    ]

    lines.append("")
    lines.append("Explanation Complete.")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text("\n".join(lines) + "\n")
    except OSError as exc:
        raise ExplainabilityError(f"Failed to write explanation report to '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def explain_model(
    config: ExplainConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
    shap_progress_callback: Callable[[str, float], None] | None = None,
) -> ExplainabilitySummary:
    """Generate SHAP and LIME explanations for the winning trained model.

    End-to-end pipeline:
        1. Identify the winning model from `train_model.py`'s
           `model_metrics.pkl` and load its fitted estimator, plus the
           test split, feature columns, and label encoder from
           `preprocess.py`'s saved artifacts.
        2. Draw a bounded random sample of the test split and compute
           SHAP values for it, automatically choosing between
           `shap.TreeExplainer` and a model-agnostic `shap.Explainer`
           fallback (see `_compute_shap_values` and the module
           docstring) -- this is the step that makes this module safe
           for a multiclass `GradientBoostingClassifier` (or any other
           winning model) without ever crashing on incompatibility.
        3. Rank features by mean absolute SHAP value, both per class and
           combined (averaged) across classes.
        4. Render global SHAP plots (a top-N bar chart, a per-class
           comparison chart, and one beeswarm plot per class).
        5. Pick one representative test-set circuit per class (the
           model's most-confident prediction for that class) and explain
           each with LIME, saving one local-explanation plot per example.
        6. Write a single plain-text `explanation_report.txt` combining
           every ranking, every LIME example, which SHAP strategy was
           used and why, and every plot's path.

    Args:
        config: Explainability configuration. Defaults to
            `ExplainConfig()` (reads from `models/` and
            `models/preprocessing/`, writes plots to
            `plots/explainability/` and the report to
            `reports/explanation_report.txt`).
        progress_callback: Optional callable invoked with short status
            strings (e.g. `print`) as major steps complete, including
            which SHAP strategy was attempted/used. If `None`, no
            progress is reported; this function itself never prints.
        shap_progress_callback: Optional callable invoked with
            (description, progress_fraction) tuples as SHAP's internal
            PermutationExplainer updates its tqdm progress bar. Used by
            the Streamlit dashboard to display a live progress bar during
            the SHAP computation phase.

    Returns:
        An `ExplainabilitySummary` with the model name and type, which
        SHAP strategy was used (and why, if a fallback occurred), the
        report path, every plot's path, both feature rankings, and every
        LIME example produced.

    Raises:
        ExplainabilityError: If required artifacts are missing or
            inconsistent, every SHAP strategy fails, LIME fails to
            produce explanations, or any output file cannot be written.
    """
    active_config = config or ExplainConfig()
    _apply_publication_style()

    models_directory = Path(active_config.models_directory)
    plots_directory = Path(active_config.plots_directory)

    model_name = _load_winning_model_name(models_directory / _MODEL_METRICS_FILENAME)
    model = _load_model(models_directory, model_name)
    model_type = _describe_model_type(model)
    data = _load_explain_data(active_config)

    if progress_callback:
        progress_callback(f"Loaded winning model '{model_name}' (type: {model_type}).")

    # --- SHAP: global + per-class feature importance -----------------
    shap_sample = _sample_test_rows(
        data.x_test, active_config.shap_sample_size, active_config.random_seed
    )
    background_sample = _build_background_sample(
        data.x_test, active_config.shap_background_size, active_config.random_seed
    )

    per_class_shap_values, shap_method, shap_fallback_reason = _compute_shap_values(
        model,
        shap_sample,
        background_sample,
        data.class_names,
        active_config.shap_fallback_max_evals,
        progress_callback,
        shap_progress_callback=shap_progress_callback,
    )

    per_class_ranking_full = {
        class_name: _rank_features_by_mean_abs_shap(values, data.feature_columns)
        for class_name, values in per_class_shap_values.items()
    }
    per_class_ranking = {
        class_name: ranking[: active_config.top_n_features]
        for class_name, ranking in per_class_ranking_full.items()
    }
    overall_ranking_full = _combine_class_rankings(per_class_shap_values, data.feature_columns)
    overall_ranking = overall_ranking_full[: active_config.top_n_features]
    overall_top_feature_names = [name for name, _ in overall_ranking]

    plot_paths: dict[str, Path] = {}

    plot_paths["shap_global_importance"] = plots_directory / "shap_global_importance.png"
    _plot_global_shap_bar(
        overall_ranking, active_config.top_n_features, plot_paths["shap_global_importance"]
    )

    plot_paths["shap_importance_by_class"] = plots_directory / "shap_importance_by_class.png"
    _plot_shap_importance_by_class(
        per_class_ranking, overall_top_feature_names, plot_paths["shap_importance_by_class"]
    )

    for class_name, values in per_class_shap_values.items():
        safe_name = class_name.lower().replace(" ", "_")
        plot_key = f"shap_beeswarm_{safe_name}"
        plot_path = plots_directory / f"shap_beeswarm_{safe_name}.png"
        _plot_shap_beeswarm(values, shap_sample, class_name, plot_path)
        plot_paths[plot_key] = plot_path

    if progress_callback:
        progress_callback("SHAP plots and rankings complete.")

    # --- LIME: local explanations for one representative circuit per class
    representatives = _select_representative_instances(model, data)

    lime_explainer = LimeTabularExplainer(
        training_data=data.x_test.to_numpy(),
        feature_names=data.feature_columns,
        class_names=data.class_names,
        mode="classification",
        random_state=active_config.random_seed,
    )

    lime_examples: list[LimeExample] = []
    probabilities = model.predict_proba(data.x_test)
    for class_name, row_index in representatives.items():
        class_index = data.class_names.index(class_name)
        row = data.x_test.iloc[row_index]

        contributions, explanation = _explain_instance_with_lime(
            lime_explainer, model, row, class_index, active_config.lime_num_features
        )

        safe_name = class_name.lower().replace(" ", "_")
        plot_key = f"lime_explanation_{safe_name}"
        plot_path = plots_directory / f"lime_explanation_{safe_name}.png"
        _plot_lime_explanation(explanation, class_name, plot_path)
        plot_paths[plot_key] = plot_path

        lime_examples.append(
            LimeExample(
                row_index=row_index,
                true_class=str(data.y_test_labels[row_index]),
                predicted_class=class_name,
                confidence=float(probabilities[row_index, class_index]),
                contributions=contributions,
                plot_path=plot_path,
            )
        )

    if progress_callback:
        progress_callback("LIME local explanations complete.")

    report_path = Path(active_config.report_path)
    _write_explanation_report(
        report_path,
        model_name,
        model_type,
        shap_method,
        shap_fallback_reason,
        overall_ranking,
        per_class_ranking,
        lime_examples,
        shap_sample_rows=len(shap_sample),
        total_test_rows=len(data.x_test),
        plot_paths=plot_paths,
    )

    if progress_callback:
        progress_callback(f"Explanation report written to '{report_path}'.")

    return ExplainabilitySummary(
        model_name=model_name,
        model_type=model_type,
        shap_method=shap_method,
        shap_fallback_reason=shap_fallback_reason,
        report_path=report_path,
        plot_paths=plot_paths,
        global_feature_ranking=overall_ranking,
        per_class_feature_ranking=per_class_ranking,
        global_feature_ranking_full=overall_ranking_full,
        per_class_feature_ranking_full=per_class_ranking_full,
        lime_examples=lime_examples,
    )


def explain_local_circuit(
    qasm_path: str | Path,
    config: ExplainConfig | None = None,
    max_features: int = 12,
    progress_callback: Callable[[str], None] | None = None,
) -> LocalShapExplanation:
    """Compute true local SHAP values for one uploaded/analyzed circuit.

    `explain_model` intentionally computes dataset-level/global SHAP
    artifacts once per trained model. This helper explains the single
    feature row built from the current circuit, so the signed SHAP
    contributions change when the user changes circuits.
    """
    from src.inference import (  # Imported lazily to avoid changing module import costs.
        _build_feature_vector,
        _build_raw_feature_dict,
        _load_circuit,
    )

    active_config = config or ExplainConfig()
    resolved_qasm_path = Path(qasm_path)
    models_directory = Path(active_config.models_directory)

    model_name = _load_winning_model_name(models_directory / _MODEL_METRICS_FILENAME)
    model = _load_model(models_directory, model_name)
    label_encoder = _load_label_encoder(Path(active_config.label_encoder_path))
    class_names = [str(label) for label in label_encoder.classes_]
    feature_columns = _load_feature_columns(Path(active_config.feature_columns_path))

    circuit = _load_circuit(resolved_qasm_path)
    raw_features, gate_counts = _build_raw_feature_dict(circuit)
    feature_row, unavailable_features = _build_feature_vector(
        raw_features, gate_counts, feature_columns
    )

    if not hasattr(model, "predict_proba"):
        raise ExplainabilityError(
            f"Model type '{_describe_model_type(model)}' exposes no 'predict_proba' method; "
            "local SHAP explanations require class probabilities."
        )

    probabilities = model.predict_proba(feature_row)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    class_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(class_names, probabilities)
    }

    data = _load_explain_data(active_config)
    background_sample = _build_background_sample(
        data.x_test, active_config.shap_background_size, active_config.random_seed
    )

    if progress_callback:
        progress_callback("Computing circuit-specific local SHAP values...")

    per_class_shap_values, shap_method, shap_fallback_reason = _compute_shap_values(
        model,
        feature_row,
        background_sample,
        class_names,
        active_config.shap_fallback_max_evals,
        progress_callback,
    )

    signed_values = per_class_shap_values[predicted_class][0]
    row_values = feature_row.iloc[0]
    order = np.argsort(np.abs(signed_values))[::-1][:max_features]
    contributions = [
        LocalShapContribution(
            feature=feature_columns[index],
            value=float(row_values.iloc[index]),
            shap_value=float(signed_values[index]),
        )
        for index in order
    ]

    return LocalShapExplanation(
        predicted_class=predicted_class,
        confidence=float(probabilities[predicted_index]),
        class_probabilities=class_probabilities,
        shap_method=shap_method,
        shap_fallback_reason=shap_fallback_reason,
        contributions=contributions,
        unavailable_features=unavailable_features,
    )


if __name__ == "__main__":
    # Demonstration / default CLI entry point: explain the winning model
    # from the standard training output location and print a short
    # summary of the top features, which SHAP strategy was used, and
    # every artifact written.
    summary = explain_model(progress_callback=print)

    print()
    print("=" * 60)
    print("Model Explanation Complete")
    print("=" * 60)
    print(f"Explained model : {summary.model_name}")
    print(f"Model type      : {summary.model_type}")
    print(f"SHAP method     : {summary.shap_method}")
    if summary.shap_fallback_reason:
        print(f"Fallback reason : {summary.shap_fallback_reason}")
    print()
    print(f"Top {min(10, len(summary.global_feature_ranking))} globally important features:")
    for rank, (feature, score) in enumerate(summary.global_feature_ranking[:10], start=1):
        print(f"  {rank:>2}. {feature:<30} {score:.5f}")
    print()
    print("LIME representative examples:")
    for example in summary.lime_examples:
        print(
            f"  {example.predicted_class:<8} (row {example.row_index}, "
            f"confidence={example.confidence:.3f}, true={example.true_class})"
        )
    print()
    print("Plots written:")
    for plot_name, plot_path in summary.plot_paths.items():
        print(f"  {plot_name:<28}: {plot_path}")
    print()
    print(f"Explanation report: {summary.report_path}")
