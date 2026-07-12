"""
preprocess.py
-------------
Single Responsibility:
    Prepare the merged training dataset (produced by
    `dataset_generator.py`) for machine learning: separate features from
    targets, drop non-ML identifier columns, encode the categorical
    reliability label, split into train/test partitions, and persist
    every artifact needed to reproduce or reuse this exact preprocessing
    later -- feature column order, the fitted label encoder, the split
    datasets themselves, and a metadata record describing how they were
    produced.

This module intentionally does NOT:
    - Train or evaluate any machine learning model
    - Engineer new features (scaling, PCA, feature selection, ...)
      beyond target separation, categorical-target encoding, and the
      train/test split -- that is deliberately left to a future
      `feature_engineering.py` / model-training module, so this
      module's output is the stable, minimal "clean tabular data"
      contract every future modeling approach can build on.
    - Parse, analyze, simulate, or generate circuits (see parser.py,
      analyzer.py, noise_simulator.py, circuit_generator.py)
    - Modify `dataset_generator.py`, `dataset_validator.py`, or `eda.py`
      in any way, or re-run any part of the generation pipeline
    - Print anything to the console outside of its own
      `if __name__ == "__main__":` demonstration block

Where this fits in the pipeline:
    Circuit Generator -> Parser -> Analyzer -> Feature Extractor
    -> Noise Simulator -> Dataset Generator -> Preprocessing
    -> Machine Learning (future)

Design summary:
    - Two ML targets are recognized from `dataset_generator.py`'s
      output schema: `reliability_class` (categorical: LOW / MEDIUM /
      HIGH -- the classification target) and `reliability_score`
      (float -- the regression target). Both are removed from the
      feature matrix and handled separately; a future model-training
      module chooses which one it needs.
    - `circuit_name` and `source_file` are row identifiers, not
      features -- they are dropped from the feature matrix entirely
      and never encoded or fed to a model.
    - The categorical target is encoded with `sklearn.preprocessing.
      LabelEncoder`, fit once on the FULL dataset (not just the
      training split) so every class label present anywhere in the
      data is guaranteed a stable integer code, then persisted with
      `joblib` so a later inference run can invert predictions back to
      LOW/MEDIUM/HIGH without re-fitting -- re-fitting on a subset
      could silently reassign integer codes if that subset's class
      distribution or first-seen ordering differed.
    - The train/test split is stratified on the encoded classification
      target, keeping the LOW/MEDIUM/HIGH proportions consistent
      across both partitions -- important here since the project's own
      EDA report shows a non-uniform class split (LOW ~50%,
      MEDIUM/HIGH ~25% each); an unstratified split could easily skew
      a small test set.
    - Every artifact needed to reproduce or audit this exact
      preprocessing later (feature column order, the fitted encoder,
      split parameters, dataset shape, class-to-code mapping) is
      persisted under `models/preprocessing/`, mirroring
      `dataset_generator.py`'s own "write everything needed to resume
      or audit a run" philosophy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Bumped whenever this module's output schema or behavior changes in a
# way future consumers (a model-training module) should know about.
# Mirrors `dataset_generator.PROJECT_VERSION`'s role for this module.
PREPROCESS_VERSION = "1.0.0"

# Identifier columns that carry no predictive signal and must never be
# fed to a model. Kept as an explicit, named constant -- rather than an
# inline literal -- so a future identifier column (e.g. a run id) can be
# added here without touching the drop logic itself.
_NON_FEATURE_COLUMNS: tuple[str, ...] = ("circuit_name", "source_file")

# The two ML targets `dataset_generator.py` produces. Both are removed
# from the feature matrix; which one a future model trains against is
# that model's decision, not this module's.
_CLASSIFICATION_TARGET_COLUMN = "reliability_class"
_REGRESSION_TARGET_COLUMN = "reliability_score"

_TARGET_COLUMNS: tuple[str, ...] = (
    _CLASSIFICATION_TARGET_COLUMN,
    _REGRESSION_TARGET_COLUMN,
)

# Default output layout under `models/`. Kept as named constants so the
# on-disk contract is easy to read in one place and to change centrally.
_DEFAULT_OUTPUT_DIRECTORY = Path("models") / "preprocessing"
_SPLITS_SUBDIRECTORY_NAME = "splits"
_FEATURE_COLUMNS_FILENAME = "feature_columns.json"
_LABEL_ENCODER_FILENAME = "label_encoder.joblib"
_METADATA_FILENAME = "preprocessing_metadata.json"


class PreprocessingError(Exception):
    """Raised when the dataset cannot be preprocessed as configured.

    Kept as a project-specific exception -- mirroring `QasmParsingError`
    (parser.py), `CircuitSimulationError` (noise_simulator.py), and
    `GenerationError` (circuit_generator.py) -- so callers can catch one
    stable error type regardless of which internal step failed (missing
    file, missing column, invalid split configuration, write failure).
    """


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for one preprocessing run.

    Attributes:
        dataset_path: Path to the merged training dataset CSV, as
            produced by `dataset_generator.generate_dataset_from_batches`
            or `dataset_generator.generate_dataset`.
        output_directory: Root directory to write every preprocessing
            artifact into (feature column list, label encoder,
            metadata, and the split CSVs under its `splits/`
            subfolder). Created if it doesn't exist.
        test_size: Fraction of rows held out for the test split
            (0.0 < test_size < 1.0).
        random_seed: Seed passed to `train_test_split` for a
            reproducible partition given the same dataset and
            `test_size`.
        stratify_on_classification_target: Whether to stratify the
            split on the encoded `reliability_class` column, keeping
            class proportions consistent across train and test. Set to
            `False` only if a target class has too few rows to support
            stratification (see `preprocess_dataset`'s docstring).
    """

    dataset_path: str | Path = Path("datasets") / "training_dataset.csv"
    output_directory: str | Path = _DEFAULT_OUTPUT_DIRECTORY
    test_size: float = 0.2
    random_seed: int | None = 42
    stratify_on_classification_target: bool = True


@dataclass
class PreprocessResult:
    """Summary of a completed preprocessing run.

    Attributes:
        feature_columns_path: Path to the saved feature column list.
        label_encoder_path: Path to the saved, fitted `LabelEncoder`.
        metadata_path: Path to the saved preprocessing metadata JSON.
        x_train_path: Path to the saved training feature matrix CSV.
        x_test_path: Path to the saved test feature matrix CSV.
        y_train_path: Path to the saved training targets CSV.
        y_test_path: Path to the saved test targets CSV.
        num_rows: Total rows in the source dataset.
        num_features: Number of feature columns retained.
        train_rows: Rows in the training split.
        test_rows: Rows in the test split.
        class_mapping: Mapping of encoded integer -> original class
            label (LOW / MEDIUM / HIGH), in encoder order.
    """

    feature_columns_path: Path
    label_encoder_path: Path
    metadata_path: Path
    x_train_path: Path
    x_test_path: Path
    y_train_path: Path
    y_test_path: Path
    num_rows: int
    num_features: int
    train_rows: int
    test_rows: int
    class_mapping: dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading and schema validation
# ---------------------------------------------------------------------------


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the merged training dataset CSV from disk.

    Args:
        dataset_path: Path to the dataset CSV.

    Returns:
        The loaded dataset as a DataFrame.

    Raises:
        PreprocessingError: If the file does not exist, is not a file,
            or is empty (zero rows).
    """
    if not dataset_path.exists():
        raise PreprocessingError(f"Dataset not found: {dataset_path}")
    if not dataset_path.is_file():
        raise PreprocessingError(f"Expected a file, got a directory: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise PreprocessingError(f"Failed to read dataset '{dataset_path}': {exc}") from exc

    if df.empty:
        raise PreprocessingError(f"Dataset '{dataset_path}' contains zero rows.")

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Verify the dataset contains every column this module depends on.

    Checks both ML target columns and every configured non-feature
    (identifier) column, since both drive downstream logic (target
    separation, encoding, dropping). Failing fast here with a single,
    complete list of what's missing is far more useful than an opaque
    `KeyError` partway through processing.

    Args:
        df: The loaded dataset.

    Raises:
        PreprocessingError: If any required column is missing, listing
            every missing column at once.
    """
    required_columns = set(_TARGET_COLUMNS) | set(_NON_FEATURE_COLUMNS)
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise PreprocessingError(
            "Dataset is missing required column(s): "
            f"{', '.join(missing_columns)}. Expected the schema produced "
            "by dataset_generator.py."
        )


# ---------------------------------------------------------------------------
# Feature / target separation
# ---------------------------------------------------------------------------


def _split_features_and_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate the feature matrix from both ML targets.

    Drops the configured identifier columns (`circuit_name`,
    `source_file`) and both target columns (`reliability_class`,
    `reliability_score`) from the feature matrix. All remaining columns
    -- structural features (depth, width, qubit counts, ...), ML
    features (gate-type counts, entangling/parameterized counts, ...),
    and noise-simulation metrics (fidelity, TVD, Hellinger distance,
    success probabilities) -- are retained as features, since this
    module makes no assumption about which of those a future model
    will find useful; that selection is left to a future
    `feature_engineering.py` / model-training module.

    Args:
        df: The full, validated dataset.

    Returns:
        A tuple of (feature matrix, raw classification target series,
        regression target series). The classification target is
        returned un-encoded (its original LOW/MEDIUM/HIGH strings);
        encoding happens separately in `_encode_classification_target`.
    """

    # Columns that would leak the answer to the ML model
    LEAKAGE_COLUMNS = [
        "estimated_fidelity",
        "success_probability_ideal",
        "success_probability_noisy",
        "total_variation_distance",
        "hellinger_distance",
    ]

    columns_to_drop = (
        list(_NON_FEATURE_COLUMNS)
        + list(_TARGET_COLUMNS)
        + LEAKAGE_COLUMNS
    )

    feature_matrix = df.drop(columns=columns_to_drop)

    classification_target = df[_CLASSIFICATION_TARGET_COLUMN].copy()
    regression_target = df[_REGRESSION_TARGET_COLUMN].copy()

    return feature_matrix, classification_target, regression_target


def _encode_classification_target(
    classification_target: pd.Series,
) -> tuple[pd.Series, LabelEncoder]:
    """Fit a `LabelEncoder` on the full classification target and apply it.

    Fitting on the full column (not a train-only subset) guarantees
    every class present anywhere in the dataset -- LOW, MEDIUM, HIGH --
    receives a stable integer code, and that the same encoder can later
    be used at inference time without risk of encountering an unseen
    label.

    Args:
        classification_target: The raw (string-labeled) classification
            target column.

    Returns:
        A tuple of (encoded target as a pandas Series of ints, the
        fitted `LabelEncoder`).
    """
    encoder = LabelEncoder()
    encoded_values = encoder.fit_transform(classification_target)
    encoded_series = pd.Series(
        encoded_values, index=classification_target.index, name=_CLASSIFICATION_TARGET_COLUMN
    )
    return encoded_series, encoder


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------


@dataclass
class _SplitData:
    """Container for one train/test split of features and both targets."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_class_train: pd.Series
    y_class_test: pd.Series
    y_reg_train: pd.Series
    y_reg_test: pd.Series


def _split_train_test(
    feature_matrix: pd.DataFrame,
    encoded_classification_target: pd.Series,
    regression_target: pd.Series,
    test_size: float,
    random_seed: int | None,
    stratify: bool,
) -> _SplitData:
    """Partition features and both targets into train and test sets.

    A single `train_test_split` call is used (rather than one call per
    target) so the same row-level partition applies consistently to
    the feature matrix and both targets -- a row's features, its
    classification label, and its regression score always land in the
    same split.

    Args:
        feature_matrix: The full feature matrix.
        encoded_classification_target: The integer-encoded
            classification target, aligned by index with
            `feature_matrix`.
        regression_target: The regression target, aligned by index
            with `feature_matrix`.
        test_size: Fraction of rows held out for testing.
        random_seed: Seed for reproducibility.
        stratify: Whether to stratify on the encoded classification
            target.

    Returns:
        A `_SplitData` with all six partitioned pieces.

    Raises:
        PreprocessingError: If `test_size` is not in (0.0, 1.0), or if
            stratification is requested but at least one class has
            fewer than 2 members (the minimum `train_test_split`
            requires per stratum).
    """
    if not 0.0 < test_size < 1.0:
        raise PreprocessingError(f"test_size must be between 0 and 1, got {test_size}")

    stratify_argument = encoded_classification_target if stratify else None

    if stratify:
        class_counts = encoded_classification_target.value_counts()
        undersized_classes = class_counts[class_counts < 2]
        if not undersized_classes.empty:
            raise PreprocessingError(
                "Cannot stratify the split: the following encoded classes "
                f"have fewer than 2 rows: {undersized_classes.to_dict()}. "
                "Set stratify_on_classification_target=False to disable "
                "stratification, or generate more data for those classes."
            )

    try:
        (
            x_train,
            x_test,
            y_class_train,
            y_class_test,
            y_reg_train,
            y_reg_test,
        ) = train_test_split(
            feature_matrix,
            encoded_classification_target,
            regression_target,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_argument,
        )
    except ValueError as exc:
        raise PreprocessingError(f"Failed to split dataset: {exc}") from exc

    return _SplitData(
        x_train=x_train,
        x_test=x_test,
        y_class_train=y_class_train,
        y_class_test=y_class_test,
        y_reg_train=y_reg_train,
        y_reg_test=y_reg_test,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_feature_columns(feature_columns: list[str], path: Path) -> None:
    """Persist the feature column names, in order, as a JSON list.

    Preserving order (not just the set of names) matters: any model
    that expects a fixed-width numeric input vector must present
    columns in exactly this order at inference time.

    Raises:
        PreprocessingError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(feature_columns, indent=2))
    except OSError as exc:
        raise PreprocessingError(f"Failed to write feature columns to '{path}': {exc}") from exc


def _save_label_encoder(encoder: LabelEncoder, path: Path) -> None:
    """Persist the fitted `LabelEncoder` with `joblib`.

    `joblib` (rather than raw `pickle`) is used since it is scikit-learn's
    own recommended serialization mechanism for fitted estimators and
    transformers, and is already a transitive dependency of scikit-learn.

    Raises:
        PreprocessingError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(encoder, path)
    except OSError as exc:
        raise PreprocessingError(f"Failed to write label encoder to '{path}': {exc}") from exc


def _save_split_csv(data: pd.DataFrame | pd.Series, path: Path, index_label: str | None = None) -> None:
    """Write one split (features or targets) to a CSV file.

    Raises:
        PreprocessingError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(path, index=False, header=True)
    except OSError as exc:
        raise PreprocessingError(f"Failed to write split file '{path}': {exc}") from exc


def _combine_targets(
    encoded_classification: pd.Series,
    raw_classification: pd.Series,
    regression: pd.Series,
) -> pd.DataFrame:
    """Assemble one targets DataFrame combining both encoded and raw labels.

    Includes the raw (string) classification label alongside its
    encoded integer form, purely for human-readable auditing of the
    saved split files -- a model-training module is expected to consume
    only the encoded column (`reliability_class`), never the raw one.

    Args:
        encoded_classification: Integer-encoded classification target,
            named `reliability_class`.
        raw_classification: The original string classification target,
            aligned by index with `encoded_classification`.
        regression: The regression target, aligned by index.

    Returns:
        A DataFrame with columns: `reliability_class` (encoded int),
        `reliability_class_label` (raw string), `reliability_score`.
    """
    return pd.DataFrame(
        {
            _CLASSIFICATION_TARGET_COLUMN: encoded_classification.reset_index(drop=True),
            f"{_CLASSIFICATION_TARGET_COLUMN}_label": raw_classification.reset_index(drop=True),
            _REGRESSION_TARGET_COLUMN: regression.reset_index(drop=True),
        }
    )


def _build_class_mapping(encoder: LabelEncoder) -> dict[int, str]:
    """Return the encoder's integer-code -> original-label mapping."""
    return {int(code): str(label) for code, label in enumerate(encoder.classes_)}


def _save_metadata(
    path: Path,
    config: PreprocessConfig,
    feature_columns: list[str],
    class_mapping: dict[int, str],
    num_rows: int,
    train_rows: int,
    test_rows: int,
) -> None:
    """Persist a JSON metadata record describing this preprocessing run.

    This is the audit trail for the whole run: what dataset was used,
    what split parameters were applied, how many features and rows
    resulted, and how classification codes map back to their original
    labels -- everything needed to sanity-check or reproduce the run
    without re-reading the (potentially large) split CSVs themselves.

    Raises:
        PreprocessingError: If the file cannot be written.
    """
    metadata: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preprocess_version": PREPROCESS_VERSION,
        "source_dataset": str(config.dataset_path),
        "num_rows": num_rows,
        "num_features": len(feature_columns),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "test_size": config.test_size,
        "random_seed": config.random_seed,
        "stratified": config.stratify_on_classification_target,
        "non_feature_columns_dropped": list(_NON_FEATURE_COLUMNS),
        "classification_target_column": _CLASSIFICATION_TARGET_COLUMN,
        "regression_target_column": _REGRESSION_TARGET_COLUMN,
        "class_mapping": class_mapping,
        "feature_columns": feature_columns,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(metadata, indent=2))
    except OSError as exc:
        raise PreprocessingError(f"Failed to write metadata to '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def preprocess_dataset(config: PreprocessConfig | None = None) -> PreprocessResult:
    """Load, clean, split, and persist the training dataset for ML use.

    End-to-end pipeline:
        1. Load `training_dataset.csv` and validate its schema.
        2. Drop identifier columns (`circuit_name`, `source_file`) and
           separate the two ML targets (`reliability_class`,
           `reliability_score`) from the feature matrix.
        3. Fit a `LabelEncoder` on the full classification target and
           encode it.
        4. Split features and both targets into train/test partitions
           (stratified on the encoded classification target, by
           default).
        5. Persist the feature column list, the fitted label encoder,
           the train/test splits, and a metadata record -- all under
           `config.output_directory`.

    Args:
        config: Preprocessing configuration. Defaults to
            `PreprocessConfig()` (reads `datasets/training_dataset.csv`,
            writes under `models/preprocessing/`, an 80/20 stratified
            split with `random_seed=42`).

    Returns:
        A `PreprocessResult` summarizing every artifact written and the
        resulting dataset shapes.

    Raises:
        PreprocessingError: If the dataset cannot be loaded, is missing
            required columns, the split configuration is invalid, or
            any output artifact cannot be written.
    """
    active_config = config or PreprocessConfig()

    dataset_path = Path(active_config.dataset_path)
    output_directory = Path(active_config.output_directory)
    splits_directory = output_directory / _SPLITS_SUBDIRECTORY_NAME

    df = _load_dataset(dataset_path)
    _validate_schema(df)

    feature_matrix, raw_classification_target, regression_target = _split_features_and_targets(df)
    encoded_classification_target, label_encoder = _encode_classification_target(
        raw_classification_target
    )

    split = _split_train_test(
        feature_matrix,
        encoded_classification_target,
        regression_target,
        test_size=active_config.test_size,
        random_seed=active_config.random_seed,
        stratify=active_config.stratify_on_classification_target,
    )

    # Re-derive each split's raw (string) classification labels by
    # position, so the saved target files can include a human-readable
    # label column alongside the encoded one (see `_combine_targets`).
    raw_train_labels = raw_classification_target.loc[split.y_class_train.index]
    raw_test_labels = raw_classification_target.loc[split.y_class_test.index]

    y_train = _combine_targets(split.y_class_train, raw_train_labels, split.y_reg_train)
    y_test = _combine_targets(split.y_class_test, raw_test_labels, split.y_reg_test)

    feature_columns_path = output_directory / _FEATURE_COLUMNS_FILENAME
    label_encoder_path = output_directory / _LABEL_ENCODER_FILENAME
    metadata_path = output_directory / _METADATA_FILENAME
    x_train_path = splits_directory / "X_train.csv"
    x_test_path = splits_directory / "X_test.csv"
    y_train_path = splits_directory / "y_train.csv"
    y_test_path = splits_directory / "y_test.csv"

    feature_columns = list(feature_matrix.columns)
    class_mapping = _build_class_mapping(label_encoder)

    _save_feature_columns(feature_columns, feature_columns_path)
    _save_label_encoder(label_encoder, label_encoder_path)
    _save_split_csv(split.x_train, x_train_path)
    _save_split_csv(split.x_test, x_test_path)
    _save_split_csv(y_train, y_train_path)
    _save_split_csv(y_test, y_test_path)
    _save_metadata(
        metadata_path,
        active_config,
        feature_columns,
        class_mapping,
        num_rows=len(df),
        train_rows=len(split.x_train),
        test_rows=len(split.x_test),
    )

    return PreprocessResult(
        feature_columns_path=feature_columns_path,
        label_encoder_path=label_encoder_path,
        metadata_path=metadata_path,
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        num_rows=len(df),
        num_features=len(feature_columns),
        train_rows=len(split.x_train),
        test_rows=len(split.x_test),
        class_mapping=class_mapping,
    )


if __name__ == "__main__":
    # Demonstration / default CLI entry point: preprocess the standard
    # dataset location with the default 80/20 stratified split, and
    # print a short summary of what was written.
    result = preprocess_dataset()

    print("=" * 60)
    print("Preprocessing Complete")
    print("=" * 60)
    print(f"Source rows       : {result.num_rows}")
    print(f"Feature columns   : {result.num_features}")
    print(f"Train rows        : {result.train_rows}")
    print(f"Test rows         : {result.test_rows}")
    print(f"Class mapping     : {result.class_mapping}")
    print()
    print("Artifacts written:")
    print(f"  Feature columns : {result.feature_columns_path}")
    print(f"  Label encoder   : {result.label_encoder_path}")
    print(f"  Metadata        : {result.metadata_path}")
    print(f"  X_train         : {result.x_train_path}")
    print(f"  X_test          : {result.x_test_path}")
    print(f"  y_train         : {result.y_train_path}")
    print(f"  y_test          : {result.y_test_path}")
