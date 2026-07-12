"""
evaluate_model.py
------------------
Single Responsibility:
    Load the winning model produced by `train_model.py`, evaluate it in
    depth on the held-out test split produced by `preprocess.py`, render
    a set of publication-quality diagnostic plots, and write a single
    human-readable evaluation report summarizing every metric.

This module intentionally does NOT:
    - Preprocess, split, or encode the dataset (see `preprocess.py`)
    - Train, retrain, or tune any model (see `train_model.py`) -- this
      module only ever calls `.predict` / `.predict_proba` on the
      already-fitted winning model, with the sole exception of the
      learning-curve plot, which clones the winning model's
      architecture and hyperparameters (never its fitted state) purely
      to trace how performance scales with training-set size
    - Modify any artifact `preprocess.py` or `train_model.py` produced
    - Print anything to the console outside of its own progress
      reporting and `if __name__ == "__main__":` summary -- the same
      documented exception `dataset_generator.py`, `circuit_generator.py`,
      and `train_model.py` make for long-running batch jobs

Where this fits in the pipeline:
    Circuit Generator -> Parser -> Analyzer -> Feature Extractor
    -> Noise Simulator -> Dataset Generator -> Preprocessing
    -> Model Training -> Model Evaluation (this module)
    -> Machine Learning consumers (future)

Design summary:
    - The winning model's identity is never hardcoded: it is read from
      `model_metrics.pkl`'s `"winning_model"` key (written by
      `train_model.py`), and the corresponding `<name>.pkl` is loaded
      from `models/`. This module therefore keeps working unchanged
      regardless of which candidate algorithm actually won.
    - Every plot is rendered with a shared "publication style" (serif-
      free, high-DPI, consistent font sizes, light grid, tight layout)
      applied once via `_apply_publication_style`, so all eight figures
      look like a single, coherent report rather than eight
      independently-styled charts.
    - ROC curves and AUC are computed one-vs-rest per class (this is a
      3-class problem: LOW / MEDIUM / HIGH), plus a micro-average
      curve, since a single binary ROC curve isn't defined for a
      multiclass target.
    - The learning curve is computed via `sklearn.model_selection.
      learning_curve` on the training split only (never touching the
      test split), using a `sklearn.base.clone` of the winning model --
      `clone` copies only the estimator's constructor parameters, never
      its fitted state, so the learning curve reflects genuinely
      refitting the same architecture at increasing training-set
      sizes, not reusing the already-trained winning model.
    - The prediction-distribution plot compares actual vs. predicted
      class counts on the test set side-by-side, which is a fast visual
      check for systematic bias toward one class (e.g. a model that
      over-predicts the majority "LOW" class the EDA report shows).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # Headless rendering: this module never opens a GUI window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder, label_binarize

# Bumped whenever this module's plot set, metrics, or output schema
# changes in a way future consumers should know about. Mirrors
# `preprocess.PREPROCESS_VERSION` / `train_model.TRAIN_MODEL_VERSION`.
EVALUATE_MODEL_VERSION = "1.0.0"

# Default locations, matching `preprocess.py`'s and `train_model.py`'s
# own output layouts exactly -- this module reads those artifacts
# verbatim rather than regenerating them.
_DEFAULT_MODELS_DIRECTORY = Path("models")
_DEFAULT_PREPROCESSING_DIRECTORY = _DEFAULT_MODELS_DIRECTORY / "preprocessing"
_DEFAULT_SPLITS_DIRECTORY = _DEFAULT_PREPROCESSING_DIRECTORY / "splits"
_DEFAULT_PLOTS_DIRECTORY = Path("plots") / "evaluation"
_DEFAULT_REPORT_PATH = Path("reports") / "evaluation_report.txt"

_CLASSIFICATION_TARGET_COLUMN = "reliability_class"
_MODEL_METRICS_FILENAME = "model_metrics.pkl"

# Publication-style figure defaults, applied once via
# `_apply_publication_style`. Kept as named constants so every figure's
# look is controlled from one place.
_FIGURE_DPI = 300
_FIGURE_FACECOLOR = "white"
_PALETTE = ("#2E5EAA", "#D65F5F", "#3CA070", "#E1A340", "#7B6FD1")


class EvaluationError(Exception):
    """Raised when model evaluation cannot proceed as configured.

    Kept as a project-specific exception -- mirroring `TrainModelError`
    (train_model.py), `PreprocessingError` (preprocess.py), and
    `GenerationError` (circuit_generator.py) -- so callers can catch one
    stable error type regardless of which internal step failed (missing
    artifacts, an unsupported model type, a write failure).
    """


@dataclass(frozen=True)
class EvaluateConfig:
    """Configuration for one evaluation run.

    Attributes:
        models_directory: Directory containing the winning model
            (`<name>.pkl`) and `model_metrics.pkl`, as written by
            `train_model.py`.
        splits_directory: Directory containing `X_train.csv`,
            `X_test.csv`, `y_train.csv`, `y_test.csv`, as written by
            `preprocess.py`.
        feature_columns_path: Path to the feature column list JSON
            written by `preprocess.py`.
        label_encoder_path: Path to the fitted `LabelEncoder` `.joblib`
            file written by `preprocess.py`.
        plots_directory: Directory to save every evaluation plot into.
            Created if it doesn't exist.
        report_path: Path to write the plain-text evaluation report to.
            Its parent directory is created if it doesn't exist.
        top_n_features: Number of top features to display on the
            feature importance plot.
        learning_curve_cv: Number of cross-validation folds used when
            computing the learning curve.
        learning_curve_train_sizes: Fractions of the training split
            used at each learning-curve step.
        random_seed: Seed used for the cloned estimator in the learning
            curve and for `learning_curve`'s own internal shuffling.
    """

    models_directory: str | Path = _DEFAULT_MODELS_DIRECTORY
    splits_directory: str | Path = _DEFAULT_SPLITS_DIRECTORY
    feature_columns_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "feature_columns.json"
    label_encoder_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "label_encoder.joblib"
    plots_directory: str | Path = _DEFAULT_PLOTS_DIRECTORY
    report_path: str | Path = _DEFAULT_REPORT_PATH
    top_n_features: int = 20
    learning_curve_cv: int = 3
    learning_curve_train_sizes: tuple[float, ...] = (0.1, 0.325, 0.55, 0.775, 1.0)
    random_seed: int | None = 42


@dataclass
class EvaluationSummary:
    """Summary of a completed evaluation run.

    Attributes:
        model_name: Name of the evaluated (winning) model.
        report_path: Path to the written evaluation report.
        plot_paths: Mapping of plot identifier -> saved file path, for
            every plot this module produces (confusion_matrix, roc,
            feature_importance, learning_curve, prediction_distribution).
        metrics: Overall test-set accuracy / precision / recall / F1
            (macro and weighted), plus per-class ROC AUC.
    """

    model_name: str
    report_path: Path
    plot_paths: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading trained artifacts
# ---------------------------------------------------------------------------


def _load_winning_model_name(model_metrics_path: Path) -> str:
    """Read which candidate won from `train_model.py`'s saved metrics file."""
    if not model_metrics_path.exists():
        raise EvaluationError(
            f"Missing model metrics at '{model_metrics_path}'. Run "
            "train_model.py before evaluate_model.py."
        )
    try:
        payload = joblib.load(model_metrics_path)
    except (OSError, EOFError) as exc:
        raise EvaluationError(
            f"Failed to load model metrics from '{model_metrics_path}': {exc}"
        ) from exc

    winning_model = payload.get("winning_model")
    if not winning_model:
        raise EvaluationError(
            f"'{model_metrics_path}' does not contain a 'winning_model' key."
        )
    return str(winning_model)


def _load_model(models_directory: Path, model_name: str) -> ClassifierMixin:
    """Load the winning model's fitted estimator from `models/<name>.pkl`."""
    model_path = models_directory / f"{model_name}.pkl"
    if not model_path.exists():
        raise EvaluationError(
            f"Winning model file not found at '{model_path}'. Run "
            "train_model.py before evaluate_model.py."
        )
    try:
        return joblib.load(model_path)
    except (OSError, EOFError) as exc:
        raise EvaluationError(f"Failed to load model from '{model_path}': {exc}") from exc


def _load_feature_columns(path: Path) -> list[str]:
    """Load the ordered feature column list saved by `preprocess.py`."""
    if not path.exists():
        raise EvaluationError(f"Missing feature columns file at '{path}'.")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Failed to read feature columns from '{path}': {exc}") from exc


def _load_label_encoder(path: Path) -> LabelEncoder:
    """Load the fitted `LabelEncoder` saved by `preprocess.py`."""
    if not path.exists():
        raise EvaluationError(f"Missing label encoder at '{path}'.")
    try:
        return joblib.load(path)
    except (OSError, EOFError) as exc:
        raise EvaluationError(f"Failed to load label encoder from '{path}': {exc}") from exc


def _load_csv(path: Path, description: str) -> pd.DataFrame:
    """Load one preprocessing split CSV, raising a clear error if absent."""
    if not path.exists():
        raise EvaluationError(f"Missing {description} at '{path}'. Run preprocess.py first.")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise EvaluationError(f"Failed to read {description} from '{path}': {exc}") from exc


@dataclass
class _EvaluationData:
    """In-memory container for everything this module needs to evaluate the model."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_columns: list[str]
    label_encoder: LabelEncoder
    class_names: list[str]


def _load_evaluation_data(config: EvaluateConfig) -> _EvaluationData:
    """Load every artifact needed to evaluate the winning model.

    Raises:
        EvaluationError: If any required file is missing, unreadable,
            or inconsistent with the saved feature column list.
    """
    splits_dir = Path(config.splits_directory)

    x_train = _load_csv(splits_dir / "X_train.csv", "training feature matrix")
    x_test = _load_csv(splits_dir / "X_test.csv", "test feature matrix")
    y_train_df = _load_csv(splits_dir / "y_train.csv", "training targets")
    y_test_df = _load_csv(splits_dir / "y_test.csv", "test targets")

    feature_columns = _load_feature_columns(Path(config.feature_columns_path))
    label_encoder = _load_label_encoder(Path(config.label_encoder_path))

    if list(x_test.columns) != feature_columns or list(x_train.columns) != feature_columns:
        raise EvaluationError(
            "Feature matrix columns do not match feature_columns.json. "
            "Re-run preprocess.py to regenerate consistent artifacts."
        )

    for name, frame in (("y_train.csv", y_train_df), ("y_test.csv", y_test_df)):
        if _CLASSIFICATION_TARGET_COLUMN not in frame.columns:
            raise EvaluationError(f"'{_CLASSIFICATION_TARGET_COLUMN}' column missing from {name}.")

    return _EvaluationData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train_df[_CLASSIFICATION_TARGET_COLUMN].to_numpy(),
        y_test=y_test_df[_CLASSIFICATION_TARGET_COLUMN].to_numpy(),
        feature_columns=feature_columns,
        label_encoder=label_encoder,
        class_names=[str(label) for label in label_encoder.classes_],
    )


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------


def _apply_publication_style() -> None:
    """Configure matplotlib rcParams for consistent, publication-quality figures.

    Applied once, before any figure is created, so every plot this
    module produces shares the same DPI, font sizes, spine styling, and
    grid appearance -- rather than each plotting function setting its
    own ad hoc style.
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
        EvaluationError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=_FIGURE_DPI, facecolor=_FIGURE_FACECOLOR)
    except OSError as exc:
        raise EvaluationError(f"Failed to write plot to '{path}': {exc}") from exc
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _compute_core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute overall accuracy, macro, and weighted precision/recall/F1."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


# ---------------------------------------------------------------------------
# Plot 1: Confusion Matrix
# ---------------------------------------------------------------------------


def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], path: Path
) -> None:
    """Render and save a confusion matrix heatmap with count annotations."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value > threshold else "#222222"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    _save_figure(fig, path)


# ---------------------------------------------------------------------------
# Plot 2: ROC Curves (one-vs-rest, multiclass)
# ---------------------------------------------------------------------------


def _plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    path: Path,
) -> dict[str, float]:
    """Render one-vs-rest ROC curves (per class + micro-average) and save them.

    Args:
        y_true: True encoded class labels for the test set.
        y_score: Predicted class probabilities, shape (n_samples, n_classes),
            in the same column order as `class_names`.
        class_names: Original class label strings, in encoder order.
        path: Destination image path.

    Returns:
        A mapping of `"auc_<class_name>"` -> AUC, plus `"auc_micro"`.
    """
    num_classes = len(class_names)
    y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    auc_scores: dict[str, float] = {}

    for class_index, class_name in enumerate(class_names):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true_binarized[:, class_index], y_score[:, class_index]
        )
        class_auc = auc(false_positive_rate, true_positive_rate)
        auc_scores[f"auc_{class_name}"] = float(class_auc)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            color=_PALETTE[class_index % len(_PALETTE)],
            linewidth=2,
            label=f"{class_name} (AUC = {class_auc:.3f})",
        )

    micro_fpr, micro_tpr, _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())
    micro_auc = auc(micro_fpr, micro_tpr)
    auc_scores["auc_micro"] = float(micro_auc)
    ax.plot(
        micro_fpr,
        micro_tpr,
        color="#444444",
        linewidth=2,
        linestyle="--",
        label=f"micro-average (AUC = {micro_auc:.3f})",
    )

    ax.plot([0, 1], [0, 1], color="#BBBBBB", linewidth=1, linestyle=":")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")

    _save_figure(fig, path)
    return auc_scores


# ---------------------------------------------------------------------------
# Plot 3: Feature Importance
# ---------------------------------------------------------------------------


def _plot_feature_importance(
    model: ClassifierMixin, feature_columns: list[str], top_n: int, path: Path
) -> None:
    """Render and save a horizontal bar chart of the top-N feature importances.

    Raises:
        EvaluationError: If the model exposes no `feature_importances_`.
    """
    if not hasattr(model, "feature_importances_"):
        raise EvaluationError(
            f"Model type '{type(model).__name__}' has no 'feature_importances_' "
            "attribute; cannot render the feature importance plot."
        )

    importance_df = (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .iloc[::-1]  # reverse so the largest bar is at the top when plotted
    )

    fig_height = max(4.0, 0.35 * len(importance_df))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.barh(importance_df["feature"], importance_df["importance"], color=_PALETTE[0])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(importance_df)} Feature Importances")

    _save_figure(fig, path)


# ---------------------------------------------------------------------------
# Plot 4: Learning Curve
# ---------------------------------------------------------------------------


def _plot_learning_curve(
    model: ClassifierMixin,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    cv: int,
    train_sizes: tuple[float, ...],
    random_seed: int | None,
    path: Path,
) -> None:
    """Render and save a learning curve (train vs. cross-validated score).

    Uses `sklearn.base.clone(model)` -- a fresh, unfitted estimator with
    the winning model's exact architecture and hyperparameters -- so
    each point is a genuine refit at that training-set size, computed
    only over the training split (the test split is never touched
    here, keeping it held out for every other evaluation in this
    module).

    Raises:
        EvaluationError: If `learning_curve` fails (e.g. `cv` too large
            for the smallest class count).
    """
    try:
        estimator = clone(model)
    except TypeError as exc:
        raise EvaluationError(f"Failed to clone model for learning curve: {exc}") from exc

    try:
        sizes, train_scores, validation_scores = learning_curve(
            estimator,
            x_train,
            y_train,
            cv=cv,
            train_sizes=np.array(train_sizes),
            scoring="f1_macro",
            n_jobs=-1,
            random_state=random_seed,
        )
    except (ValueError, TypeError) as exc:
        raise EvaluationError(f"Failed to compute learning curve: {exc}") from exc

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_mean, color=_PALETTE[0], marker="o", linewidth=2, label="Training score")
    ax.fill_between(
        sizes, train_mean - train_std, train_mean + train_std, color=_PALETTE[0], alpha=0.15
    )
    ax.plot(
        sizes,
        validation_mean,
        color=_PALETTE[1],
        marker="o",
        linewidth=2,
        label="Cross-validation score",
    )
    ax.fill_between(
        sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        color=_PALETTE[1],
        alpha=0.15,
    )

    ax.set_xlabel("Training Examples")
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Learning Curve")
    ax.legend(loc="lower right")

    _save_figure(fig, path)


# ---------------------------------------------------------------------------
# Plot 5: Prediction Distribution
# ---------------------------------------------------------------------------


def _plot_prediction_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], path: Path
) -> None:
    """Render and save a grouped bar chart of actual vs. predicted class counts.

    A fast visual check for systematic bias -- e.g. a model that
    over-predicts the majority class relative to its true frequency.
    """
    num_classes = len(class_names)
    actual_counts = np.bincount(y_true, minlength=num_classes)
    predicted_counts = np.bincount(y_pred, minlength=num_classes)

    x_positions = np.arange(num_classes)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(
        x_positions - bar_width / 2,
        actual_counts,
        width=bar_width,
        color=_PALETTE[0],
        label="Actual",
    )
    ax.bar(
        x_positions + bar_width / 2,
        predicted_counts,
        width=bar_width,
        color=_PALETTE[1],
        label="Predicted",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Count")
    ax.set_title("Actual vs. Predicted Class Distribution (Test Set)")
    ax.legend()

    for positions, counts in ((x_positions - bar_width / 2, actual_counts),
                              (x_positions + bar_width / 2, predicted_counts)):
        for x_pos, count in zip(positions, counts):
            ax.text(x_pos, count, str(int(count)), ha="center", va="bottom", fontsize=9)

    _save_figure(fig, path)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _write_evaluation_report(
    path: Path,
    model_name: str,
    core_metrics: dict[str, float],
    auc_scores: dict[str, float],
    classification_report_text: str,
    class_names: list[str],
    num_train_rows: int,
    num_test_rows: int,
    num_features: int,
    plot_paths: dict[str, Path],
) -> None:
    """Write the plain-text evaluation report summarizing every metric and plot.

    Raises:
        EvaluationError: If the file cannot be written.
    """
    separator = "=" * 70
    lines: list[str] = [
        separator,
        "Quantum-Reliability-AI -- Model Evaluation Report",
        separator,
        "",
        f"Evaluated Model    : {model_name}",
        f"Classes            : {', '.join(class_names)}",
        f"Training Rows      : {num_train_rows}",
        f"Test Rows          : {num_test_rows}",
        f"Feature Count      : {num_features}",
        "",
        "-" * 70,
        "Overall Metrics (Test Set)",
        "-" * 70,
        f"Accuracy                    : {core_metrics['accuracy']:.4f}",
        f"Precision (macro)           : {core_metrics['precision_macro']:.4f}",
        f"Recall (macro)              : {core_metrics['recall_macro']:.4f}",
        f"F1 Score (macro)            : {core_metrics['f1_macro']:.4f}",
        f"Precision (weighted)        : {core_metrics['precision_weighted']:.4f}",
        f"Recall (weighted)           : {core_metrics['recall_weighted']:.4f}",
        f"F1 Score (weighted)         : {core_metrics['f1_weighted']:.4f}",
        "",
        "-" * 70,
        "ROC AUC (One-vs-Rest)",
        "-" * 70,
    ]

    for class_name in class_names:
        lines.append(f"{class_name:<20}: {auc_scores[f'auc_{class_name}']:.4f}")
    lines.append(f"{'micro-average':<20}: {auc_scores['auc_micro']:.4f}")

    lines += [
        "",
        "-" * 70,
        "Per-Class Classification Report",
        "-" * 70,
        classification_report_text.rstrip(),
        "",
        "-" * 70,
        "Generated Plots",
        "-" * 70,
    ]
    for plot_name, plot_path in plot_paths.items():
        lines.append(f"{plot_name:<24}: {plot_path}")

    lines.append("")
    lines.append("Evaluation Complete.")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text("\n".join(lines) + "\n")
    except OSError as exc:
        raise EvaluationError(f"Failed to write evaluation report to '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_model(config: EvaluateConfig | None = None) -> EvaluationSummary:
    """Evaluate the winning trained model and produce a full diagnostic report.

    End-to-end pipeline:
        1. Identify the winning model from `train_model.py`'s
           `model_metrics.pkl` and load its fitted estimator.
        2. Load the test (and, for the learning curve, training) split,
           feature column list, and fitted label encoder from
           `preprocess.py`'s saved artifacts.
        3. Compute accuracy, precision, recall, F1 (macro and weighted),
           and per-class ROC AUC on the test set.
        4. Render five publication-quality plots: confusion matrix, ROC
           curves, feature importance, learning curve, and prediction
           distribution -- all saved under `config.plots_directory`.
        5. Write a single plain-text `evaluation_report.txt` summarizing
           every metric and listing every generated plot's path.

    Args:
        config: Evaluation configuration. Defaults to `EvaluateConfig()`
            (reads from `models/` and `models/preprocessing/`, writes
            plots to `plots/evaluation/` and the report to
            `reports/evaluation_report.txt`).

    Returns:
        An `EvaluationSummary` with the model name, report path, every
        plot's path, and the computed metrics.

    Raises:
        EvaluationError: If required artifacts are missing or
            inconsistent, the model lacks `predict_proba` or
            `feature_importances_`, or any output file cannot be
            written.
    """
    active_config = config or EvaluateConfig()
    _apply_publication_style()

    models_directory = Path(active_config.models_directory)
    plots_directory = Path(active_config.plots_directory)

    model_name = _load_winning_model_name(models_directory / _MODEL_METRICS_FILENAME)
    model = _load_model(models_directory, model_name)
    data = _load_evaluation_data(active_config)

    if not hasattr(model, "predict_proba"):
        raise EvaluationError(
            f"Model type '{type(model).__name__}' has no 'predict_proba' method; "
            "cannot compute ROC curves."
        )

    y_pred = model.predict(data.x_test)
    y_score = model.predict_proba(data.x_test)

    core_metrics = _compute_core_metrics(data.y_test, y_pred)
    classification_report_text = classification_report(
        data.y_test, y_pred, target_names=data.class_names, zero_division=0
    )

    plot_paths: dict[str, Path] = {
        "confusion_matrix": plots_directory / "confusion_matrix.png",
        "roc_curve": plots_directory / "roc_curve.png",
        "feature_importance": plots_directory / "feature_importance.png",
        "learning_curve": plots_directory / "learning_curve.png",
        "prediction_distribution": plots_directory / "prediction_distribution.png",
    }

    _plot_confusion_matrix(data.y_test, y_pred, data.class_names, plot_paths["confusion_matrix"])
    auc_scores = _plot_roc_curves(data.y_test, y_score, data.class_names, plot_paths["roc_curve"])
    _plot_feature_importance(
        model, data.feature_columns, active_config.top_n_features, plot_paths["feature_importance"]
    )
    _plot_learning_curve(
        model,
        data.x_train,
        data.y_train,
        active_config.learning_curve_cv,
        active_config.learning_curve_train_sizes,
        active_config.random_seed,
        plot_paths["learning_curve"],
    )
    _plot_prediction_distribution(
        data.y_test, y_pred, data.class_names, plot_paths["prediction_distribution"]
    )

    report_path = Path(active_config.report_path)
    _write_evaluation_report(
        report_path,
        model_name,
        core_metrics,
        auc_scores,
        classification_report_text,
        data.class_names,
        num_train_rows=len(data.x_train),
        num_test_rows=len(data.x_test),
        num_features=len(data.feature_columns),
        plot_paths=plot_paths,
    )

    return EvaluationSummary(
        model_name=model_name,
        report_path=report_path,
        plot_paths=plot_paths,
        metrics={**core_metrics, **auc_scores},
    )


if __name__ == "__main__":
    # Demonstration / default CLI entry point: evaluate the winning
    # model from the standard training output location and print a
    # short summary of the metrics and every artifact written.
    summary = evaluate_model()

    print("=" * 60)
    print("Model Evaluation Complete")
    print("=" * 60)
    print(f"Evaluated model : {summary.model_name}")
    print()
    print("Core metrics:")
    for metric_name in (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ):
        print(f"  {metric_name:<20}: {summary.metrics[metric_name]:.4f}")
    print()
    print("Plots written:")
    for plot_name, plot_path in summary.plot_paths.items():
        print(f"  {plot_name:<24}: {plot_path}")
    print()
    print(f"Evaluation report: {summary.report_path}")
