"""
dataset_generator.py
---------------------
Single Responsibility:
    Orchestrate the existing pipeline (parser -> analyzer ->
    feature_extractor -> noise_simulator) over an entire directory of
    OpenQASM 3 circuits, merge each circuit's results into one flat
    record, and export the collection as a machine-learning-ready CSV
    dataset plus run metadata.

This module intentionally does NOT:
    - Parse, analyze, extract features from, or simulate circuits itself
      (it calls the existing modules for every one of those steps)
    - Train or run any machine learning model
    - Modify parser.py, analyzer.py, feature_extractor.py,
      noise_simulator.py, or circuit_generator.py in any way

Where this fits in the pipeline:
    Circuit Generator -> Parser -> Analyzer -> Feature Extractor
    -> Noise Simulator -> Dataset Generator -> Machine Learning (future)

Design summary (fault-tolerant persistence, see module-level helpers):
    - `generate_dataset()` still processes one directory of `.qasm` files
      and keeps that directory's records in memory only for the
      duration of the call (this is the "one batch" unit referred to
      below) -- its public behavior and return type are unchanged.
    - `generate_dataset_from_batches()` is the resumable, crash-safe
      entry point. It processes one batch folder at a time by calling
      `generate_dataset()` unmodified, and only *after* a batch folder
      has completed successfully does it:
        1. Append that batch's rows onto the single top-level
           `training_dataset.csv` (never rewriting rows already
           written).
        2. Append that batch's failures onto the single top-level
           `error_log.csv`.
        3. Rewrite `checkpoint.json` and `dataset_metadata.json` to
           reflect progress through that batch.
      If the process is interrupted (Ctrl+C, power loss, ...) while a
      batch is still being processed, nothing for that batch has been
      appended yet, so no partial/duplicate rows are ever produced --
      the batch simply restarts, in full, the next run.
    - On startup, `generate_dataset_from_batches()` reads
      `checkpoint.json` (if present) and resumes from
      `last_completed_batch + 1`, skipping every batch already
      recorded as complete.
    - A circuit failure is caught, logged, and does not stop the run.
"""

from __future__ import annotations

import csv
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from qiskit_aer.noise import NoiseModel

from src.analyzer import analyze_circuit
from src.feature_extractor import extract_features
from src.noise_simulator import (
    DEFAULT_SHOTS,
    CircuitSimulationError,
    build_noise_model,
    simulate_noise,
)
from src.parser import QasmParsingError, load_qasm_file

# Bumped whenever this module's output schema or generation behavior
# changes in a way future consumers (ml_trainer.py) should know about.
# Not read from anywhere else in the project; kept local and explicit
# since the project has no centralized version source yet.
PROJECT_VERSION = "0.5.0"

# Base (non-gate) columns, in the order they're written to the CSV.
# Kept as an explicit ordered list -- rather than relying on dict
# insertion order alone -- so the schema is easy to read and diff.
_BASE_COLUMNS: tuple[str, ...] = (
    "circuit_name",
    "source_file",
    "number_of_qubits",
    "number_of_classical_bits",
    "depth",
    "width",
    "total_operations",
    "single_qubit_gates",
    "two_qubit_gates",
    "three_qubit_gates",
    "measurement_gates",
    "parameterized_gates",
    "entangling_gates",
    "estimated_fidelity",
    "total_variation_distance",
    "hellinger_distance",
    "success_probability_ideal",
    "success_probability_noisy",
    "reliability_class",
    "reliability_score",
)

_ERROR_LOG_COLUMNS: tuple[str, ...] = ("source_file", "error_type", "error_message")

_INTERRUPT_MESSAGE = (
    "Dataset generation interrupted.\n"
    "Completed batches are already saved.\n"
    "Resume later by running\n\n"
    "python3 -m src.dataset_generator\n\n"
    "again."
)


@dataclass
class DatasetGenerationResult:
    """Summary of a completed (or partial) dataset generation run.

    Attributes:
        dataset_path: Path to the final `training_dataset.csv`.
        metadata_path: Path to `dataset_metadata.json`.
        error_log_path: Path to `error_log.csv` (present even if empty).
        total_circuits: Total `.qasm` files discovered.
        successful_circuits: Circuits successfully processed.
        failed_circuits: Circuits that raised an error.
    """

    dataset_path: Path
    metadata_path: Path
    error_log_path: Path
    total_circuits: int
    successful_circuits: int
    failed_circuits: int


@dataclass
class BatchedDatasetGenerationResult:
    """Summary of a run over every batch folder under a batches root directory.

    Attributes:
        dataset_path: Path to the single merged `training_dataset.csv`.
        metadata_path: Path to the merged `dataset_metadata.json`.
        error_log_path: Path to the single merged `error_log.csv`.
        total_batches: Number of batch folders discovered and processed.
        total_circuits: Total `.qasm` files across all batches.
        successful_circuits: Circuits successfully processed, all batches.
        failed_circuits: Circuits that raised an error, all batches.
        generation_seconds: Wall-clock time for the entire multi-batch run,
            summed across every session (a resumed run adds to the time
            already recorded in `checkpoint.json`, it does not reset it).
    """

    dataset_path: Path
    metadata_path: Path
    error_log_path: Path
    total_batches: int
    total_circuits: int
    successful_circuits: int
    failed_circuits: int
    generation_seconds: float


@dataclass
class _FailureRecord:
    """One entry in the in-memory error log."""

    source_file: str
    error_type: str
    error_message: str


def _discover_qasm_files(input_directory: Path) -> list[Path]:
    """Return every `.qasm` file in `input_directory`, sorted by name.

    Sorted so processing order (and therefore checkpoint contents) is
    deterministic across runs given the same directory contents.
    """
    return sorted(input_directory.glob("*.qasm"))


def _process_circuit_file(
    qasm_path: Path, shots: int, noise_model: NoiseModel
) -> dict[str, Any]:
    """Run one circuit through the full existing pipeline and merge results.

    Calls `parser.load_qasm_file`, `analyzer.analyze_circuit`,
    `feature_extractor.extract_features`, and `noise_simulator.simulate_noise`
    -- in that order -- and merges their outputs into one flat record.
    The record's `gate_distribution` field is left as a nested dict; it
    is flattened later, once, against the full run's gate vocabulary
    (see `_flatten_record`).

    Args:
        qasm_path: Path to the `.qasm` file to process.
        shots: Shot count to pass to `simulate_noise`.
        noise_model: NoiseModel to pass to `simulate_noise`.

    Returns:
        A flat (except for `gate_distribution`) record dictionary.

    Raises:
        FileNotFoundError, ValueError, QasmParsingError: From parsing.
        CircuitSimulationError: From noise simulation.
        Exception: Any other unexpected failure from the underlying
            pipeline modules is allowed to propagate; the caller is
            responsible for catching and logging it (see
            `generate_dataset`), so one bad circuit never stops the run.
    """
    qc = load_qasm_file(qasm_path)
    analysis = analyze_circuit(qc)
    features = extract_features(qc)
    noise_result = simulate_noise(qc, shots=shots, noise_model=noise_model)

    return {
        "circuit_name": analysis["name"] or qasm_path.stem,
        "source_file": qasm_path.name,
        "number_of_qubits": analysis["num_qubits"],
        "number_of_classical_bits": analysis["num_clbits"],
        "depth": analysis["depth"],
        "width": analysis["width"],
        "total_operations": analysis["total_operations"],
        "single_qubit_gates": features["single_qubit_gates"],
        "two_qubit_gates": features["two_qubit_gates"],
        "three_qubit_gates": features["three_qubit_gates"],
        "measurement_gates": features["measurement_gates"],
        "parameterized_gates": features["parameterized_gates"],
        "entangling_gates": features["entangling_gates"],
        "estimated_fidelity": noise_result["estimated_fidelity"],
        "total_variation_distance": noise_result["total_variation_distance"],
        "hellinger_distance": noise_result["hellinger_distance"],
        "success_probability_ideal": noise_result["ideal_success_probability"],
        "success_probability_noisy": noise_result["noisy_success_probability"],
        "reliability_class": noise_result["circuit_reliability"],
        "reliability_score": noise_result["estimated_reliability_percent"],
        "gate_distribution": analysis["gate_counts"],
    }


def _flatten_records(
    records: list[dict[str, Any]], gate_vocabulary: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Flatten every record's nested `gate_distribution` into gate_* columns.

    Re-flattens from scratch against the full current vocabulary every
    time it's called, so every returned row has the same column set --
    including gate columns for gate types that only appear in circuits
    processed after a given row was first created.

    Args:
        records: Raw records as produced by `_process_circuit_file`.
        gate_vocabulary: Every gate name seen anywhere in the run so far.

    Returns:
        A tuple of (flattened rows, ordered gate column names).
    """
    gate_columns = sorted(f"gate_{name}" for name in gate_vocabulary)

    flattened: list[dict[str, Any]] = []
    for record in records:
        gate_counts: dict[str, int] = record["gate_distribution"]
        row = {key: record[key] for key in _BASE_COLUMNS}
        for gate_name in gate_vocabulary:
            row[f"gate_{gate_name}"] = gate_counts.get(gate_name, 0)
        flattened.append(row)

    return flattened, gate_columns


def _write_dataset_csv(
    records: list[dict[str, Any]], gate_vocabulary: set[str], path: Path
) -> None:
    """Flatten and write the current records to a CSV file at `path`."""
    flattened, gate_columns = _flatten_records(records, gate_vocabulary)
    fieldnames = list(_BASE_COLUMNS) + gate_columns

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def _write_error_log(failures: list[_FailureRecord], path: Path) -> None:
    """Write the collected failure records to a CSV error log.

    Always writes the file (with header only) even if `failures` is
    empty, so downstream tooling can rely on the file's existence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_ERROR_LOG_COLUMNS))
        writer.writeheader()
        for failure in failures:
            writer.writerow(
                {
                    "source_file": failure.source_file,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                }
            )


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _report_progress(
    processed: int, total: int, elapsed_seconds: float
) -> None:
    """Print a single progress line: counts, average time, ETA."""
    average = elapsed_seconds / processed if processed else 0.0
    remaining = total - processed
    eta = average * remaining
    print(
        f"Processed {processed} / {total} "
        f"| Avg: {average:.3f}s/circuit "
        f"| Est. Remaining: {_format_duration(eta)}"
    )


def generate_dataset(
    input_directory: str | Path,
    output_directory: str | Path = "datasets",
    batch_size: int = 500,
    shots: int = DEFAULT_SHOTS,
    noise_model: NoiseModel | None = None,
    random_seed: int | None = None,
    progress_every: int = 1,
) -> DatasetGenerationResult:
    """Process every .qasm file in a directory into an ML-ready dataset.

    For each `.qasm` file: parse -> analyze -> extract features ->
    simulate noise, merge the results into one record, and accumulate.
    Writes a rolling checkpoint CSV every `batch_size` circuits, and the
    final dataset, error log, and metadata once all files are processed.

    This function's records for `input_directory` are kept in memory
    only for the duration of this call -- it represents the "one batch"
    unit of work described in the module docstring. Callers that need
    crash-safe, resumable persistence across *many* such directories
    should use `generate_dataset_from_batches`, which calls this
    function once per batch folder and only persists a batch's rows
    once this function returns successfully for that folder.

    Args:
        input_directory: Directory containing `.qasm` files (e.g. the
            output of `circuit_generator.generate_circuits`).
        output_directory: Directory to write `training_dataset.csv`,
            `training_dataset.partial.csv`, `error_log.csv`, and
            `dataset_metadata.json` into. Created if missing.
        batch_size: Number of circuits between checkpoint saves.
        shots: Shot count passed to every `simulate_noise` call.
        noise_model: Shared NoiseModel used for every circuit. Defaults
            to `noise_simulator.build_noise_model()`, built once and
            reused across all circuits (rather than once per circuit)
            purely as a performance optimization -- it does not change
            `noise_simulator.py` or duplicate its model-building logic.
        random_seed: Recorded in `dataset_metadata.json` for
            reproducibility bookkeeping. NOTE: `noise_simulator.simulate_noise`
            does not currently accept a simulator seed, so this value is
            *not* wired into the underlying Aer simulation's randomness --
            it is reported honestly as metadata only, not as a guarantee
            of reproducible noise sampling.
        progress_every: Print a progress line every N circuits processed
            (successes and failures both count).

    Returns:
        A `DatasetGenerationResult` summarizing the run.

    Raises:
        FileNotFoundError: If `input_directory` does not exist.
        ValueError: If `input_directory` is not a directory, or
            `batch_size` is not positive.
    """
    input_path = Path(input_directory)
    output_path = Path(output_directory)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Expected a directory, got a file: {input_path}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    active_noise_model = noise_model if noise_model is not None else build_noise_model()
    noise_model_description = (
        "custom user-supplied NoiseModel"
        if noise_model is not None
        else "default synthetic NoiseModel (see noise_simulator.build_noise_model)"
    )

    qasm_files = _discover_qasm_files(input_path)
    total_circuits = len(qasm_files)

    records: list[dict[str, Any]] = []
    failures: list[_FailureRecord] = []
    gate_vocabulary: set[str] = set()

    dataset_path = output_path / "training_dataset.csv"
    partial_path = output_path / "training_dataset.partial.csv"
    error_log_path = output_path / "error_log.csv"
    metadata_path = output_path / "dataset_metadata.json"

    start_time = time.monotonic()

    try:
        for processed_count, qasm_path in enumerate(qasm_files, start=1):
            try:
                record = _process_circuit_file(qasm_path, shots, active_noise_model)
                records.append(record)
                gate_vocabulary.update(record["gate_distribution"].keys())
            except (
                FileNotFoundError,
                ValueError,
                QasmParsingError,
                CircuitSimulationError,
                TypeError,
            ) as exc:
                failures.append(
                    _FailureRecord(
                        source_file=qasm_path.name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- orchestrator must never stop the run
                failures.append(
                    _FailureRecord(
                        source_file=qasm_path.name,
                        error_type=type(exc).__name__,
                        error_message=f"{exc} | {traceback.format_exc(limit=1)}",
                    )
                )

            if progress_every and processed_count % progress_every == 0:
                _report_progress(processed_count, total_circuits, time.monotonic() - start_time)

            if processed_count % batch_size == 0:
                _write_dataset_csv(records, gate_vocabulary, partial_path)
    except KeyboardInterrupt:
        # Preserve whatever the last intra-directory checkpoint captured
        # (written every `batch_size` circuits above) so this directory's
        # progress isn't lost even though it hasn't been promoted to a
        # completed batch yet. The caller (generate_dataset_from_batches)
        # is responsible for deciding this directory is *not* complete --
        # it simply won't see a returned result to append.
        # Deliberately silent here (no user-facing message, no clean
        # exit): this function's own caller decides what "interrupted"
        # means for it. `generate_dataset_from_batches` -- the resumable
        # entry point -- prints the interruption message and exits
        # cleanly itself; a caller using `generate_dataset` standalone
        # sees a normal KeyboardInterrupt propagate, as it did before.
        if records:
            _write_dataset_csv(records, gate_vocabulary, partial_path)
        raise

    # Final progress line, in case total_circuits isn't a multiple of progress_every.
    if total_circuits and total_circuits % max(progress_every, 1) != 0:
        _report_progress(total_circuits, total_circuits, time.monotonic() - start_time)

    _write_dataset_csv(records, gate_vocabulary, dataset_path)
    _write_error_log(failures, error_log_path)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_circuits": total_circuits,
        "successful_circuits": len(records),
        "failed_circuits": len(failures),
        "noise_model": noise_model_description,
        "shots": shots,
        "random_seed": random_seed,
        "project_version": PROJECT_VERSION,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # The rolling checkpoint is superseded by the final dataset once the
    # run completes successfully; remove it so it isn't mistaken for a
    # second, possibly-stale artifact.
    if partial_path.exists():
        partial_path.unlink()

    return DatasetGenerationResult(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        error_log_path=error_log_path,
        total_circuits=total_circuits,
        successful_circuits=len(records),
        failed_circuits=len(failures),
    )


# ---------------------------------------------------------------------------
# Persistence helpers for the resumable, batch-folder-level API below.
#
# These are the only pieces of this module concerned with *how* progress
# survives a crash: incremental CSV appends, an incremental error log,
# and an atomically-written checkpoint + metadata pair. `generate_dataset`
# above (and the pipeline it calls) is untouched by any of this.
# ---------------------------------------------------------------------------


def _discover_batch_directories(batches_root: Path) -> list[Path]:
    """Return every immediate subdirectory of `batches_root`, sorted by name.

    Sorted so batch processing order is deterministic across runs. Any
    non-directory entries (stray files) directly under `batches_root` are
    ignored.
    """
    return sorted(p for p in batches_root.iterdir() if p.is_dir())


def _remove_directory_tree(path: Path) -> None:
    """Recursively remove a directory tree, ignoring a non-existent path."""
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (write-to-temp then rename).

    Used for `checkpoint.json` and `dataset_metadata.json` so a crash
    mid-write can never leave either file half-written / unparsable --
    the rename is atomic on the same filesystem, so readers only ever
    see the old complete file or the new complete file, never a partial
    one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load `checkpoint.json`, or return None if absent, empty, or corrupt.

    A corrupt or unreadable checkpoint is treated the same as a missing
    one (start over from batch 1) rather than raising, since a half
    -written checkpoint from an *extremely* unlucky crash should never
    be able to crash the *next* run too.
    """
    if not checkpoint_path.exists():
        return None
    try:
        return json.loads(checkpoint_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_checkpoint(checkpoint_path: Path, checkpoint_data: dict[str, Any]) -> None:
    """Atomically rewrite `checkpoint.json` with the latest progress."""
    _atomic_write_text(checkpoint_path, json.dumps(checkpoint_data, indent=2))


def _update_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    """Atomically rewrite `dataset_metadata.json` with the latest progress."""
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2))


def _append_error_rows(source_error_log_path: Path, error_log_path: Path) -> int:
    """Append one completed batch's error rows onto the merged error log.

    The error log's column set (`_ERROR_LOG_COLUMNS`) never changes
    batch-to-batch, so this is a plain streamed append -- no schema
    reconciliation is ever needed here (unlike `_append_training_rows`).

    Args:
        source_error_log_path: A single batch's own `error_log.csv`, as
            written by `generate_dataset` for that batch folder.
        error_log_path: The single, top-level, ever-growing error log.

    Returns:
        The number of rows appended.
    """
    if not source_error_log_path.exists():
        return 0

    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    file_is_new = not error_log_path.exists()

    rows_appended = 0
    with source_error_log_path.open("r", newline="") as src_f:
        reader = csv.DictReader(src_f)
        with error_log_path.open("a", newline="") as dst_f:
            writer = csv.DictWriter(dst_f, fieldnames=list(_ERROR_LOG_COLUMNS))
            if file_is_new:
                writer.writeheader()
            for row in reader:
                writer.writerow(row)
                rows_appended += 1

    return rows_appended


def _append_training_rows(
    source_dataset_path: Path,
    dataset_path: Path,
    known_gate_columns: set[str],
) -> tuple[int, set[str]]:
    """Append one completed batch's rows onto the merged training dataset.

    Never overwrites a row already written. Three cases:

    1. `dataset_path` doesn't exist yet -- create it, header included,
       using this batch's own gate columns.
    2. `dataset_path` exists and this batch introduces no gate columns
       the merged file doesn't already have -- a plain streamed append
       (the common case after the first batch or two, once the gate
       vocabulary has stabilized).
    3. `dataset_path` exists but this batch uses one or more gate
       columns not yet in the merged file's header -- the merged file
       is streamed, row-by-row, into a temp file under the *widened*
       header (existing rows get 0 for the new columns), this batch's
       rows are appended to that same temp file, and the temp file
       atomically replaces the old one. This is the only case where
       previously-written rows are rewritten (to add columns, never to
       change values), and it is bounded by an atomic rename so a crash
       mid-migration leaves the original, complete file untouched.

    Args:
        source_dataset_path: A single batch's own `training_dataset.csv`.
        dataset_path: The single, top-level, ever-growing training dataset.
        known_gate_columns: Every `gate_*` column name seen in the merged
            file so far (tracked in `checkpoint.json` across runs, since
            an in-memory set doesn't survive a restart).

    Returns:
        A tuple of (rows appended, updated known gate column set).
    """
    with source_dataset_path.open("r", newline="") as src_f:
        reader = csv.DictReader(src_f)
        source_header = reader.fieldnames or []
        source_gate_columns = {c for c in source_header if c.startswith("gate_")}
        updated_gate_columns = known_gate_columns | source_gate_columns

        if not dataset_path.exists():
            fieldnames = list(_BASE_COLUMNS) + sorted(updated_gate_columns)
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            rows_appended = 0
            with dataset_path.open("w", newline="") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames, restval=0)
                writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    rows_appended += 1
            return rows_appended, updated_gate_columns

        with dataset_path.open("r", newline="") as existing_f:
            existing_header = next(csv.reader(existing_f), [])
        existing_gate_columns = {c for c in existing_header if c.startswith("gate_")}

        if source_gate_columns <= existing_gate_columns:
            # No new columns: plain append under the existing header.
            rows_appended = 0
            with dataset_path.open("a", newline="") as out_f:
                writer = csv.DictWriter(
                    out_f, fieldnames=existing_header, restval=0, extrasaction="ignore"
                )
                for row in reader:
                    writer.writerow(row)
                    rows_appended += 1
            return rows_appended, existing_gate_columns | source_gate_columns

        # New gate columns: migrate the existing file to a widened header,
        # then append this batch's rows, all in one streamed pass, then
        # atomically swap it in.
        widened_gate_columns = existing_gate_columns | source_gate_columns
        fieldnames = list(_BASE_COLUMNS) + sorted(widened_gate_columns)
        tmp_path = dataset_path.with_suffix(dataset_path.suffix + ".migrating.tmp")

        rows_appended = 0
        with dataset_path.open("r", newline="") as old_f, tmp_path.open(
            "w", newline=""
        ) as tmp_f:
            old_reader = csv.DictReader(old_f)
            writer = csv.DictWriter(tmp_f, fieldnames=fieldnames, restval=0)
            writer.writeheader()
            for row in old_reader:
                writer.writerow(row)
            for row in reader:
                writer.writerow(row)
                rows_appended += 1

        os.replace(tmp_path, dataset_path)
        return rows_appended, widened_gate_columns


def generate_dataset_from_batches(
    batches_root_directory: str | Path = "generated_batches",
    output_directory: str | Path = "datasets",
    per_batch_checkpoint_size: int = 500,
    shots: int = DEFAULT_SHOTS,
    noise_model: NoiseModel | None = None,
    random_seed: int | None = None,
    progress_every: int = 50,
) -> BatchedDatasetGenerationResult:
    """Process every batch folder under `batches_root_directory` into one dataset.

    Auto-detects every immediate subdirectory of `batches_root_directory`
    (e.g. `batch_0001/`, `batch_0002/`, ... as produced by
    `circuit_generator.generate_dataset_batches`) and processes each one
    in order by calling the existing, unmodified `generate_dataset`
    against it.

    Crash-safe / resumable behavior:
        - After each batch folder finishes successfully, its rows are
          immediately appended to `datasets/training_dataset.csv`, its
          failures to `datasets/error_log.csv`, and `checkpoint.json` /
          `dataset_metadata.json` are rewritten -- all before moving on
          to the next batch. Nothing is held back until the whole run
          finishes.
        - If `checkpoint.json` already exists in `output_directory`
          when this function is called, processing automatically
          resumes at `last_completed_batch + 1`; every batch already
          recorded as complete is skipped, never reprocessed.
        - If interrupted (Ctrl+C, power loss, etc.) while a batch is
          still in progress, that batch has not been appended anywhere
          yet, so it simply restarts from scratch next run -- batches
          1..N-1 remain exactly as they were, and no duplicate or
          partial rows are ever produced for batch N.
        - Records for at most one batch folder are ever held in memory
          at a time (inside the `generate_dataset` call for that
          folder); once a batch is appended, that memory is freed.

    Args:
        batches_root_directory: Directory containing one subfolder per
            batch, each holding `.qasm` files.
        output_directory: Directory containing (and to write)
            `training_dataset.csv`, `error_log.csv`,
            `dataset_metadata.json`, and `checkpoint.json`.
        per_batch_checkpoint_size: `batch_size` passed through to each
            per-batch `generate_dataset` call (its intra-batch checkpoint
            interval, not related to how the batch folders themselves are
            organized).
        shots: Shot count passed to every `simulate_noise` call.
        noise_model: Shared NoiseModel used for every circuit in every
            batch. Built once here (via `noise_simulator.build_noise_model`
            if not supplied) and passed to every per-batch call, so every
            batch is simulated under the identical noise model.
        random_seed: Recorded in the merged `dataset_metadata.json` for
            reproducibility bookkeeping (same caveat as `generate_dataset`:
            not wired into Aer's own simulation randomness).
        progress_every: Passed through to each per-batch `generate_dataset`
            call as its `progress_every`.

    Returns:
        A `BatchedDatasetGenerationResult` summarizing the whole run
        (all batches, including any completed in earlier, resumed
        sessions).

    Raises:
        FileNotFoundError: If `batches_root_directory` does not exist.
        ValueError: If it exists but contains no subdirectories.
    """
    batches_root = Path(batches_root_directory)
    output_path = Path(output_directory)

    if not batches_root.exists():
        raise FileNotFoundError(f"Batches root directory not found: {batches_root}")

    batch_directories = _discover_batch_directories(batches_root)
    if not batch_directories:
        raise ValueError(f"No batch subdirectories found under: {batches_root}")

    active_noise_model = noise_model if noise_model is not None else build_noise_model()
    noise_model_description = (
        "custom user-supplied NoiseModel"
        if noise_model is not None
        else "default synthetic NoiseModel (see noise_simulator.build_noise_model)"
    )

    working_root = output_path / ".batch_runs"
    total_batches = len(batch_directories)

    dataset_path = output_path / "training_dataset.csv"
    error_log_path = output_path / "error_log.csv"
    metadata_path = output_path / "dataset_metadata.json"
    checkpoint_path = output_path / "checkpoint.json"

    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is not None:
        start_batch_index = int(checkpoint.get("last_completed_batch", 0)) + 1
        rows_written = int(checkpoint.get("rows_written", 0))
        successful_circuits = int(checkpoint.get("successful_circuits", 0))
        failed_circuits = int(checkpoint.get("failed_circuits", 0))
        previous_elapsed_seconds = float(checkpoint.get("generation_time_seconds", 0.0))
        gate_vocabulary: set[str] = set(checkpoint.get("gate_vocabulary", []))
        if start_batch_index > 1:
            print(
                f"Resuming from checkpoint: batches "
                f"1-{start_batch_index - 1} already completed."
            )
    else:
        start_batch_index = 1
        rows_written = 0
        successful_circuits = 0
        failed_circuits = 0
        previous_elapsed_seconds = 0.0
        gate_vocabulary = set()

    session_start = time.monotonic()

    def _current_elapsed() -> float:
        return previous_elapsed_seconds + (time.monotonic() - session_start)

    if start_batch_index > total_batches:
        print("All batches already completed per checkpoint; nothing to do.")
    else:
        try:
            for batch_index in range(start_batch_index, total_batches + 1):
                batch_dir = batch_directories[batch_index - 1]
                print(f"Batch {batch_index}/{total_batches} ({batch_dir.name})")

                batch_output_dir = working_root / batch_dir.name
                result = generate_dataset(
                    input_directory=batch_dir,
                    output_directory=batch_output_dir,
                    batch_size=per_batch_checkpoint_size,
                    shots=shots,
                    noise_model=active_noise_model,
                    random_seed=random_seed,
                    progress_every=progress_every,
                )

                # Only reached if the batch finished without raising --
                # this is the "batch completed successfully" boundary
                # that everything below is conditioned on.
                rows_appended, gate_vocabulary = _append_training_rows(
                    result.dataset_path, dataset_path, gate_vocabulary
                )
                _append_error_rows(result.error_log_path, error_log_path)
                _remove_directory_tree(batch_output_dir)

                rows_written += rows_appended
                successful_circuits += result.successful_circuits
                failed_circuits += result.failed_circuits
                elapsed_seconds = _current_elapsed()

                _write_checkpoint(
                    checkpoint_path,
                    {
                        "last_completed_batch": batch_index,
                        "rows_written": rows_written,
                        "successful_circuits": successful_circuits,
                        "failed_circuits": failed_circuits,
                        "generation_time_seconds": elapsed_seconds,
                        "gate_vocabulary": sorted(gate_vocabulary),
                    },
                )
                _update_metadata(
                    metadata_path,
                    {
                        "completed_batches": batch_index,
                        "total_batches": total_batches,
                        "successful_circuits": successful_circuits,
                        "failed_circuits": failed_circuits,
                        "rows_written": rows_written,
                        "elapsed_time_seconds": elapsed_seconds,
                        "last_completed_batch": batch_index,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "noise_model": noise_model_description,
                        "shots": shots,
                        "random_seed": random_seed,
                        "project_version": PROJECT_VERSION,
                    },
                )

                print(f"Batch {batch_index}/{total_batches} completed.")
                print(f"{rows_written} rows safely saved.")
                print()
        except KeyboardInterrupt:
            print()
            print(_INTERRUPT_MESSAGE)
            raise SystemExit(0)

    generation_seconds = _current_elapsed()
    total_circuits = successful_circuits + failed_circuits

    print(f"Total batches processed : {total_batches}")
    print(f"Total circuits processed: {total_circuits}")
    print(f"Successful circuits     : {successful_circuits}")
    print(f"Failed circuits         : {failed_circuits}")
    print(f"Dataset path            : {dataset_path}")
    print(f"Generation time         : {_format_duration(generation_seconds)}")

    return BatchedDatasetGenerationResult(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        error_log_path=error_log_path,
        total_batches=total_batches,
        total_circuits=total_circuits,
        successful_circuits=successful_circuits,
        failed_circuits=failed_circuits,
        generation_seconds=generation_seconds,
    )


if __name__ == "__main__":
    generate_dataset_from_batches(
        batches_root_directory="generated_batches",
        output_directory="datasets",
    )
