"""
train_model.py
---------------
Single Responsibility:
    Train several candidate classifiers on the preprocessed
    `reliability_class` target, evaluate each on the held-out test
    split produced by `preprocess.py`, automatically select the
    best-performing candidate, and persist the winning model plus every
    artifact needed to audit, explain, or reuse it.

This module intentionally does NOT:
    - Preprocess, split, or encode the dataset itself (see
      `preprocess.py`, whose saved artifacts under
      `models/preprocessing/` are this module's only input)
    - Perform hyperparameter search / tuning -- each candidate is
      trained with one fixed, reasonable configuration; tuning the
      winning candidate further is deliberately left to a future
      `tune_model.py`, so this module's job stays a clean "which
      off-the-shelf algorithm family fits best" comparison
    - Parse, analyze, simulate, or generate circuits, or touch anything
      upstream of `preprocess.py`
    - Print anything to the console outside of its own progress
      reporting and `if __name__ == "__main__":` summary -- documented,
      same exception `dataset_generator.py` and `circuit_generator.py`
      make for long-running, multi-step batch jobs

Where this fits in the pipeline:
    Circuit Generator -> Parser -> Analyzer -> Feature Extractor
    -> Noise Simulator -> Dataset Generator -> Preprocessing
    -> Model Training (this module) -> Machine Learning consumers (future)

Design summary:
    - Candidates: Random Forest, Gradient Boosting, and Extra Trees
      (all in scikit-learn, always available), plus XGBoost -- included
      only if the `xgboost` package is importable. A missing optional
      dependency is treated as "one fewer candidate to compare", never
      as a hard failure, since the module's job is to make the best
      choice among whatever is actually installed.
    - Every candidate is trained on the exact same train split and
      scored on the exact same test split (both read verbatim from
      `preprocess.py`'s saved CSVs), so comparisons are apples-to-apples.
    - Selection metric: macro-averaged F1 score, with accuracy as a
      tie-breaker. Macro F1 (rather than plain accuracy) is used as the
      primary criterion because the project's own EDA report shows a
      non-uniform class split (LOW ~50%, MEDIUM/HIGH ~25% each) --
      macro F1 weights all three classes equally instead of letting the
      majority class dominate the score.
    - The winning model is saved under a filename derived from its own
      candidate name (e.g. `random_forest.pkl`, `xgboost.pkl`) rather
      than a fixed name, so the saved artifact is always self-describing
      about which algorithm actually won.
    - Feature importances, the classification report, and the confusion
      matrix are all computed for the winning model only (not every
      candidate), since those are the artifacts a downstream consumer
      actually needs to explain and validate the model that will be
      used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True

except Exception:
    XGBClassifier = None
    _XGBOOST_AVAILABLE = False

# Bumped whenever this module's candidate set, selection metric, or
# output schema changes in a way future consumers should know about.
# Mirrors `dataset_generator.PROJECT_VERSION` / `preprocess.PREPROCESS_VERSION`.
TRAIN_MODEL_VERSION = "1.0.0"

# Default locations. `_DEFAULT_SPLITS_DIRECTORY` and
# `_DEFAULT_PREPROCESSING_DIRECTORY` must match `preprocess.py`'s own
# output layout (`models/preprocessing/` with a `splits/` subfolder) --
# this module reads those artifacts verbatim rather than regenerating
# them.
_DEFAULT_PREPROCESSING_DIRECTORY = Path("models") / "preprocessing"
_DEFAULT_SPLITS_DIRECTORY = _DEFAULT_PREPROCESSING_DIRECTORY / "splits"
_DEFAULT_OUTPUT_DIRECTORY = Path("models")

_CLASSIFICATION_TARGET_COLUMN = "reliability_class"

_FEATURE_IMPORTANCE_FILENAME = "feature_importance.csv"
_MODEL_METRICS_FILENAME = "model_metrics.pkl"
_CLASSIFICATION_REPORT_FILENAME = "classification_report.txt"
_CONFUSION_MATRIX_FILENAME = "confusion_matrix.npy"
_TRAINING_METADATA_FILENAME = "training_metadata.json"


class TrainModelError(Exception):
    """Raised when model training or evaluation cannot proceed as configured.

    Kept as a project-specific exception -- mirroring `PreprocessingError`
    (preprocess.py), `CircuitSimulationError` (noise_simulator.py), and
    `GenerationError` (circuit_generator.py) -- so callers can catch one
    stable error type regardless of which internal step failed (missing
    preprocessing artifacts, a candidate that fails to fit, a write
    failure).
    """


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for one multi-candidate training run.

    Attributes:
        splits_directory: Directory containing `X_train.csv`,
            `X_test.csv`, `y_train.csv`, `y_test.csv`, as written by
            `preprocess.preprocess_dataset`.
        feature_columns_path: Path to the feature column list JSON
            written by `preprocess.py`.
        label_encoder_path: Path to the fitted `LabelEncoder` `.joblib`
            file written by `preprocess.py`.
        output_directory: Root directory to write every training
            artifact into (the winning model, feature importances,
            metrics, classification report, confusion matrix, and
            training metadata).
        random_seed: Seed passed to every candidate estimator that
            accepts one, for reproducible training given the same
            input data.
    """

    splits_directory: str | Path = _DEFAULT_SPLITS_DIRECTORY
    feature_columns_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "feature_columns.json"
    label_encoder_path: str | Path = _DEFAULT_PREPROCESSING_DIRECTORY / "label_encoder.joblib"
    output_directory: str | Path = _DEFAULT_OUTPUT_DIRECTORY
    random_seed: int | None = 42


@dataclass
class ModelMetrics:
    """Evaluation metrics for one trained candidate on the test split.

    Attributes:
        accuracy: Overall test-set accuracy.
        precision_macro: Macro-averaged precision across classes.
        recall_macro: Macro-averaged recall across classes.
        f1_macro: Macro-averaged F1 score across classes (the primary
            model-selection criterion -- see module docstring).
        f1_weighted: Support-weighted F1 score, retained as a secondary,
            class-imbalance-aware reference metric alongside the macro
            score.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a plain, JSON/pickle-friendly dict."""
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
        }


@dataclass
class TrainingSummary:
    """Summary of a completed multi-candidate training run.

    Attributes:
        winning_model_name: Registry key of the best-performing
            candidate (e.g. "random_forest", "xgboost").
        winning_model_path: Path to the saved winning model.
        feature_importance_path: Path to the saved feature importance CSV.
        model_metrics_path: Path to the saved all-candidates metrics pickle.
        classification_report_path: Path to the saved classification
            report text file (winning model only).
        confusion_matrix_path: Path to the saved confusion matrix
            `.npy` file (winning model only).
        metadata_path: Path to the saved training metadata JSON.
        all_metrics: Every candidate's `ModelMetrics`, keyed by
            candidate name, including candidates that did not win.
        skipped_candidates: Names of candidates that were configured
            but skipped (currently only "xgboost", if the package isn't
            installed).
    """

    winning_model_name: str
    winning_model_path: Path
    feature_importance_path: Path
    model_metrics_path: Path
    classification_report_path: Path
    confusion_matrix_path: Path
    metadata_path: Path
    all_metrics: dict[str, ModelMetrics] = field(default_factory=dict)
    skipped_candidates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading preprocessing artifacts
# ---------------------------------------------------------------------------


def _load_csv(path: Path, description: str) -> pd.DataFrame:
    """Load one preprocessing split CSV, raising a clear error if absent."""
    if not path.exists():
        raise TrainModelError(
            f"Missing {description} at '{path}'. Run preprocess.py "
            "before train_model.py."
        )
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise TrainModelError(f"Failed to read {description} from '{path}': {exc}") from exc


def _load_feature_columns(path: Path) -> list[str]:
    """Load the ordered feature column list saved by `preprocess.py`."""
    if not path.exists():
        raise TrainModelError(
            f"Missing feature columns file at '{path}'. Run preprocess.py first."
        )
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainModelError(f"Failed to read feature columns from '{path}': {exc}") from exc


def _load_label_encoder(path: Path) -> LabelEncoder:
    """Load the fitted `LabelEncoder` saved by `preprocess.py`."""
    if not path.exists():
        raise TrainModelError(
            f"Missing label encoder at '{path}'. Run preprocess.py first."
        )
    try:
        return joblib.load(path)
    except (OSError, EOFError) as exc:
        raise TrainModelError(f"Failed to load label encoder from '{path}': {exc}") from exc


@dataclass
class _TrainingData:
    """In-memory container for the loaded train/test features and targets."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_columns: list[str]
    label_encoder: LabelEncoder


def _load_training_data(config: TrainConfig) -> _TrainingData:
    """Load every artifact `preprocess.py` produced that this module needs.

    Args:
        config: The active training configuration.

    Returns:
        A `_TrainingData` bundling features, targets, feature column
        order, and the fitted label encoder.

    Raises:
        TrainModelError: If any required file is missing or unreadable,
            or if the loaded feature matrix's columns don't match the
            saved feature column list.
    """
    splits_dir = Path(config.splits_directory)

    x_train = _load_csv(splits_dir / "X_train.csv", "training feature matrix")
    x_test = _load_csv(splits_dir / "X_test.csv", "test feature matrix")
    y_train_df = _load_csv(splits_dir / "y_train.csv", "training targets")
    y_test_df = _load_csv(splits_dir / "y_test.csv", "test targets")

    feature_columns = _load_feature_columns(Path(config.feature_columns_path))
    label_encoder = _load_label_encoder(Path(config.label_encoder_path))

    if list(x_train.columns) != feature_columns or list(x_test.columns) != feature_columns:
        raise TrainModelError(
            "Feature matrix columns do not match the saved feature_columns.json. "
            "Re-run preprocess.py to regenerate consistent artifacts."
        )

    if _CLASSIFICATION_TARGET_COLUMN not in y_train_df.columns:
        raise TrainModelError(
            f"'{_CLASSIFICATION_TARGET_COLUMN}' column missing from y_train.csv."
        )
    if _CLASSIFICATION_TARGET_COLUMN not in y_test_df.columns:
        raise TrainModelError(
            f"'{_CLASSIFICATION_TARGET_COLUMN}' column missing from y_test.csv."
        )

    return _TrainingData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train_df[_CLASSIFICATION_TARGET_COLUMN].to_numpy(),
        y_test=y_test_df[_CLASSIFICATION_TARGET_COLUMN].to_numpy(),
        feature_columns=feature_columns,
        label_encoder=label_encoder,
    )


# ---------------------------------------------------------------------------
# Candidate model registry
# ---------------------------------------------------------------------------


def _build_candidate_factories(
    random_seed: int | None,
) -> tuple[dict[str, Callable[[], ClassifierMixin]], list[str]]:
    """Build the registry of candidate model factories to train and compare.

    Each factory is a zero-argument callable that returns a fresh,
    unfitted estimator -- kept as factories (rather than pre-built
    instances) so every candidate starts from a clean, never-before-fit
    state regardless of call order.

    Args:
        random_seed: Seed threaded into every candidate that accepts a
            `random_state` argument.

    Returns:
        A tuple of (factory registry keyed by candidate name, list of
        candidate names that were configured but skipped because their
        backing package isn't installed).
    """
    factories: dict[str, Callable[[], ClassifierMixin]] = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_seed,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_seed,
            n_jobs=-1,
        ),
    }

    skipped: list[str] = []
    if _XGBOOST_AVAILABLE:
        factories["xgboost"] = lambda: XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_seed,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    else:
        skipped.append("xgboost")

    return factories, skipped


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    """Compute the standard metric set for one candidate's test predictions."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return ModelMetrics(
        accuracy=float(accuracy),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
    )


@dataclass
class _CandidateResult:
    """One trained candidate's fitted estimator and test-set metrics."""

    name: str
    model: ClassifierMixin
    metrics: ModelMetrics
    test_predictions: np.ndarray


def _train_candidate(
    name: str,
    factory: Callable[[], ClassifierMixin],
    data: _TrainingData,
    progress_callback: Callable[[str], None] | None,
) -> _CandidateResult:
    """Train one candidate and evaluate it on the held-out test split.

    Args:
        name: Candidate's registry name (used for logging and, if it
            wins, its saved filename).
        factory: Zero-argument callable producing a fresh estimator.
        data: The loaded training data bundle.
        progress_callback: Optional callable invoked with a short status
            string before and after training, so a caller (e.g. this
            module's own `__main__` block) can report progress without
            this function printing directly itself.

    Returns:
        A `_CandidateResult` with the fitted model, its test metrics,
        and its raw test-set predictions (reused later for the
        classification report and confusion matrix if it wins).

    Raises:
        TrainModelError: If the candidate fails to fit or predict.
    """
    if progress_callback:
        progress_callback(f"Training {name}...")

    model = factory()
    try:
        model.fit(data.x_train, data.y_train)
        predictions = model.predict(data.x_test)
    except Exception as exc:  # noqa: BLE001 -- one bad candidate must not stop the run
        raise TrainModelError(f"Candidate '{name}' failed to train or predict: {exc}") from exc

    metrics = _evaluate_predictions(data.y_test, predictions)

    if progress_callback:
        progress_callback(
            f"{name} done -- accuracy={metrics.accuracy:.4f}, f1_macro={metrics.f1_macro:.4f}"
        )

    return _CandidateResult(name=name, model=model, metrics=metrics, test_predictions=predictions)


def _select_best_candidate(results: list[_CandidateResult]) -> _CandidateResult:
    """Pick the best candidate by macro F1, breaking ties on accuracy.

    Args:
        results: Every successfully trained candidate's result.

    Returns:
        The winning `_CandidateResult`.

    Raises:
        TrainModelError: If `results` is empty (every candidate failed).
    """
    if not results:
        raise TrainModelError("No candidate model trained successfully; nothing to select.")

    return max(results, key=lambda r: (r.metrics.f1_macro, r.metrics.accuracy))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _model_filename(model_name: str) -> str:
    """Return the on-disk filename for a given candidate name (e.g. `.pkl`)."""
    return f"{model_name}.pkl"


def _save_model(model: ClassifierMixin, path: Path) -> None:
    """Persist the winning model with `joblib`.

    Raises:
        TrainModelError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, path)
    except OSError as exc:
        raise TrainModelError(f"Failed to write model to '{path}': {exc}") from exc


def _save_feature_importance(
    model: ClassifierMixin, feature_columns: list[str], path: Path
) -> None:
    """Persist the winning model's feature importances as a ranked CSV.

    Every candidate in this module's registry (Random Forest, Gradient
    Boosting, Extra Trees, XGBoost) exposes `.feature_importances_`
    after fitting, so no fallback path is needed here; a future
    candidate without that attribute would need one.

    Args:
        model: The fitted winning model.
        feature_columns: Feature names, in the same order the model was
            trained on.
        path: Destination CSV path.

    Raises:
        TrainModelError: If the model has no `feature_importances_`
            attribute, or the file cannot be written.
    """
    if not hasattr(model, "feature_importances_"):
        raise TrainModelError(
            f"Winning model type '{type(model).__name__}' has no "
            "'feature_importances_' attribute; cannot write feature_importance.csv."
        )

    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": importances}
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance_df.insert(0, "rank", range(1, len(importance_df) + 1))

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        importance_df.to_csv(path, index=False)
    except OSError as exc:
        raise TrainModelError(f"Failed to write feature importance to '{path}': {exc}") from exc


def _save_model_metrics(
    all_metrics: dict[str, ModelMetrics], winning_model_name: str, path: Path
) -> None:
    """Persist every candidate's metrics, plus the winner's name, via joblib.

    Saved as a plain dict (not a DataFrame) so it can be loaded without
    a pandas dependency by any future lightweight consumer.

    Raises:
        TrainModelError: If the file cannot be written.
    """
    payload: dict[str, Any] = {
        "winning_model": winning_model_name,
        "metrics_by_model": {name: metrics.as_dict() for name, metrics in all_metrics.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(payload, path)
    except OSError as exc:
        raise TrainModelError(f"Failed to write model metrics to '{path}': {exc}") from exc


def _save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str], path: Path
) -> None:
    """Persist a text classification report for the winning model's test predictions.

    Raises:
        TrainModelError: If the file cannot be written.
    """
    report_text = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(report_text)
    except OSError as exc:
        raise TrainModelError(f"Failed to write classification report to '{path}': {exc}") from exc


def _save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, path: Path
) -> None:
    """Persist the winning model's confusion matrix as a `.npy` array.

    `labels=range(num_classes)` is passed explicitly so the matrix's
    row/column order always matches the label encoder's class order
    (0..num_classes-1), even if the test split happens not to contain
    every class.

    Raises:
        TrainModelError: If the file cannot be written.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(path, matrix)
    except OSError as exc:
        raise TrainModelError(f"Failed to write confusion matrix to '{path}': {exc}") from exc


def _save_training_metadata(
    path: Path,
    config: TrainConfig,
    winning_model_name: str,
    all_metrics: dict[str, ModelMetrics],
    skipped_candidates: list[str],
    num_train_rows: int,
    num_test_rows: int,
    num_features: int,
) -> None:
    """Persist a JSON audit record describing this training run.

    Mirrors `preprocess.py`'s `preprocessing_metadata.json`: everything
    needed to sanity-check or reproduce the run without re-reading the
    (potentially large) model or split files themselves.

    Raises:
        TrainModelError: If the file cannot be written.
    """
    metadata: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_model_version": TRAIN_MODEL_VERSION,
        "random_seed": config.random_seed,
        "num_train_rows": num_train_rows,
        "num_test_rows": num_test_rows,
        "num_features": num_features,
        "candidates_trained": list(all_metrics.keys()),
        "candidates_skipped": skipped_candidates,
        "winning_model": winning_model_name,
        "selection_metric": "f1_macro (tie-broken by accuracy)",
        "metrics_by_model": {name: metrics.as_dict() for name, metrics in all_metrics.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(metadata, indent=2))
    except OSError as exc:
        raise TrainModelError(f"Failed to write training metadata to '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def train_models(
    config: TrainConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> TrainingSummary:
    """Train, evaluate, and select among every available candidate model.

    End-to-end pipeline:
        1. Load `preprocess.py`'s saved train/test splits, feature
           column list, and fitted label encoder.
        2. Build the candidate registry (Random Forest, Gradient
           Boosting, Extra Trees, and XGBoost if installed).
        3. Train and evaluate every candidate on the identical
           train/test split.
        4. Select the best candidate by macro F1 (tie-broken by
           accuracy).
        5. Persist the winning model, its feature importances, every
           candidate's metrics, the winning model's classification
           report and confusion matrix, and a training metadata record
           -- all under `config.output_directory`.

    Args:
        config: Training configuration. Defaults to `TrainConfig()`
            (reads from `models/preprocessing/`, writes to `models/`).
        progress_callback: Optional callable invoked with short status
            strings as training proceeds (e.g. `print`). If `None`, no
            progress is reported; this function itself never prints.

    Returns:
        A `TrainingSummary` describing the winning model and every
        artifact written.

    Raises:
        TrainModelError: If required preprocessing artifacts are
            missing or inconsistent, every candidate fails to train, or
            any output artifact cannot be written.
    """
    active_config = config or TrainConfig()
    output_directory = Path(active_config.output_directory)

    data = _load_training_data(active_config)
    factories, skipped_candidates = _build_candidate_factories(active_config.random_seed)

    if not factories:
        raise TrainModelError(
            "No candidate models are available to train (all backing "
            "packages are missing)."
        )

    results: list[_CandidateResult] = []
    for name, factory in factories.items():
        results.append(_train_candidate(name, factory, data, progress_callback))

    winner = _select_best_candidate(results)
    all_metrics = {result.name: result.metrics for result in results}

    if progress_callback:
        progress_callback(
            f"Selected '{winner.name}' as the best model "
            f"(f1_macro={winner.metrics.f1_macro:.4f}, accuracy={winner.metrics.accuracy:.4f})."
        )

    model_path = output_directory / _model_filename(winner.name)
    feature_importance_path = output_directory / _FEATURE_IMPORTANCE_FILENAME
    model_metrics_path = output_directory / _MODEL_METRICS_FILENAME
    classification_report_path = output_directory / _CLASSIFICATION_REPORT_FILENAME
    confusion_matrix_path = output_directory / _CONFUSION_MATRIX_FILENAME
    metadata_path = output_directory / _TRAINING_METADATA_FILENAME

    target_names = [str(label) for label in data.label_encoder.classes_]
    num_classes = len(target_names)

    _save_model(winner.model, model_path)
    _save_feature_importance(winner.model, data.feature_columns, feature_importance_path)
    _save_model_metrics(all_metrics, winner.name, model_metrics_path)
    _save_classification_report(
        data.y_test, winner.test_predictions, target_names, classification_report_path
    )
    _save_confusion_matrix(
        data.y_test, winner.test_predictions, num_classes, confusion_matrix_path
    )
    _save_training_metadata(
        metadata_path,
        active_config,
        winner.name,
        all_metrics,
        skipped_candidates,
        num_train_rows=len(data.x_train),
        num_test_rows=len(data.x_test),
        num_features=len(data.feature_columns),
    )

    return TrainingSummary(
        winning_model_name=winner.name,
        winning_model_path=model_path,
        feature_importance_path=feature_importance_path,
        model_metrics_path=model_metrics_path,
        classification_report_path=classification_report_path,
        confusion_matrix_path=confusion_matrix_path,
        metadata_path=metadata_path,
        all_metrics=all_metrics,
        skipped_candidates=skipped_candidates,
    )


if __name__ == "__main__":
    # Demonstration / default CLI entry point: train and compare every
    # available candidate against the standard preprocessing output
    # location, printing progress as each candidate finishes and a
    # final summary of the winner and every artifact written.
    summary = train_models(progress_callback=print)

    print()
    print("=" * 60)
    print("Model Training Complete")
    print("=" * 60)
    print(f"Winning model      : {summary.winning_model_name}")
    if summary.skipped_candidates:
        print(f"Skipped candidates : {', '.join(summary.skipped_candidates)} (not installed)")
    print()
    print("Metrics by candidate:")
    for candidate_name, metrics in summary.all_metrics.items():
        marker = " <-- selected" if candidate_name == summary.winning_model_name else ""
        print(
            f"  {candidate_name:<20} accuracy={metrics.accuracy:.4f} "
            f"f1_macro={metrics.f1_macro:.4f} f1_weighted={metrics.f1_weighted:.4f}{marker}"
        )
    print()
    print("Artifacts written:")
    print(f"  Model                  : {summary.winning_model_path}")
    print(f"  Feature importance     : {summary.feature_importance_path}")
    print(f"  Model metrics          : {summary.model_metrics_path}")
    print(f"  Classification report  : {summary.classification_report_path}")
    print(f"  Confusion matrix       : {summary.confusion_matrix_path}")
    print(f"  Training metadata      : {summary.metadata_path}")
