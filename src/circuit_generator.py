"""
circuit_generator.py
---------------------
Single Responsibility:
    Generate a diverse, reproducible collection of valid QuantumCircuit
    objects, export each as an OpenQASM 3 (.qasm) file, and record
    structural metadata for every generated circuit -- at both small
    demonstration scale and large (50,000+) research-dataset scale.

This module intentionally does NOT:
    - Assemble a labeled machine learning dataset (pairing features with
      simulated reliability targets is the job of `dataset_generator.py`)
    - Train or run any machine learning model
    - Simulate noise or estimate reliability (see `noise_simulator.py`)
    - Print anything to the console outside of its own `if __name__ ==
      "__main__":` demonstration block and the per-batch statistics
      printed by `generate_dataset_batches` (documented as its one
      intentional exception -- see that function's docstring)

Where this fits in the pipeline:
    Circuit Generator -> Parser -> Analyzer -> Feature Extractor
    -> Noise Simulator -> Dataset Generator -> Machine Learning (future)

Two public entry points:
    - `generate_circuits`: the original small/demo-scale API. Writes a
      single flat directory of circuits + one metadata.csv. Kept
      unchanged in behavior for backward compatibility.
    - `generate_dataset_batches`: the new large-scale engine. Writes
      `output_directory/batch_NNNN/` subfolders, each containing its own
      `.qasm` files and `metadata.csv` -- so each batch folder is, on its
      own, exactly the flat-directory shape `dataset_generator.py`
      already expects. No changes to `dataset_generator.py` are needed;
      point it at one batch folder at a time.

Design summary (see accompanying review/explanation for full reasoning):
    - Circuits are generated from `CircuitArchetype` profiles, which are
      either structural-style biases (low-depth, sparse, heavy
      entanglement, ...) or named templates with real canonical
      structure (Bell state, GHZ state, QFT, variational ansatz).
      Assignment to circuit slots uses weighted round-robin (computed
      fresh per batch), so every batch -- not just the whole dataset --
      has guaranteed, balanced family coverage.
    - Circuits are built layer-by-layer (gates assigned to disjoint,
      still-free qubits within a layer) so the emergent circuit depth
      tracks the requested target depth; a `max_gates` ceiling bounds
      total gate count regardless of depth.
    - A coarse structural signature (num_qubits, depth, gate_counts),
      hashed with BLAKE2b, is used to detect and retry near-duplicate
      circuits. Only the fixed-size hash is retained across the run
      (not the full signature), keeping memory bounded even across many
      batches.
    - At batch scale, only one QuantumCircuit object is held in memory
      at a time: it's generated, analyzed, written to disk, and
      released before the next one is built. Only scalar running totals
      (sums/counts) are retained for the printed statistics.
    - Metadata reuses `analyzer.analyze_circuit` and
      `feature_extractor.extract_features` rather than recomputing
      structural facts, avoiding duplicated analysis logic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.qasm3 import dumps as qasm3_dumps

from src.analyzer import analyze_circuit
from src.feature_extractor import extract_features

# Maximum attempts to regenerate a circuit that collides (by structural
# signature hash) with one already produced in this run, before accepting
# the duplicate and moving on. Kept small: this is a cheap diversity
# nudge, not a correctness guarantee.
_MAX_DUPLICATE_RETRIES = 5

# Angle sampling range for parameterized gates (radians). Uniform over a
# full rotation gives even coverage of Bloch-sphere rotations for
# RX/RY/RZ and full phase coverage for CP, without biasing the generator
# toward any particular noise-sensitivity region.
_ANGLE_RANGE: tuple[float, float] = (0.0, 2 * math.pi)

# Default gate pools. Kept as module-level constants (rather than
# embedded in generation logic) so a new gate can be supported by adding
# one entry here, without touching the layer-building logic itself.
# Archetypes may override these per-family (see `CircuitArchetype`).
_FIXED_SINGLE_QUBIT_GATES: tuple[str, ...] = ("h", "x", "y", "z", "s", "t", "sx")
_PARAMETERIZED_SINGLE_QUBIT_GATES: tuple[str, ...] = ("rx", "ry", "rz")
_FIXED_TWO_QUBIT_GATES: tuple[str, ...] = ("cx", "cz", "swap")
_PARAMETERIZED_TWO_QUBIT_GATES: tuple[str, ...] = ("cp",)
_THREE_QUBIT_GATES: tuple[str, ...] = ("ccx",)

# Single-qubit gate pool restricted to Clifford-group generators (used by
# the "random_clifford" family). Excludes T (the pi/8 gate is the
# canonical non-Clifford gate) and any parameterized rotation. SX (sqrt-X)
# and S/S-dagger ARE Clifford, so they're included.
_CLIFFORD_SINGLE_QUBIT_GATES: tuple[str, ...] = ("h", "x", "y", "z", "s", "sdg", "sx")
# Two-qubit Clifford generators. CP is excluded: it's only Clifford for
# angles that are multiples of pi/2, and this generator samples continuous
# angles, so it's simplest and most correct to exclude it entirely here.
_CLIFFORD_TWO_QUBIT_GATES: tuple[str, ...] = ("cx", "cz", "swap")

# Probability (within a layer, once a >=3-qubit entangling placement has
# already been decided) of choosing a three-qubit gate over a two-qubit
# one. Kept low and fixed across archetypes: three-qubit gates are
# structurally rare in most real circuits.
_THREE_QUBIT_GATE_PROBABILITY = 0.15

# Approximate number of circuit layers consumed by one variational-ansatz
# block (rotation layer + entangling ring), used to translate a sampled
# target depth into a number of ansatz repetitions.
_VARIATIONAL_LAYERS_PER_BLOCK = 3


class GenerationError(Exception):
    """Raised when circuit generation cannot proceed as configured.

    Kept as a project-specific exception -- mirroring `QasmParsingError`
    (parser.py) and `CircuitSimulationError` (noise_simulator.py) -- so
    callers can catch one stable error type regardless of which internal
    step failed (invalid configuration, filesystem failure, export
    failure).
    """


@dataclass(frozen=True)
class CircuitArchetype:
    """A named generation profile: either a structural-style bias or a
    named template with real canonical structure.

    Attributes:
        name: Short identifier, stored in metadata as `circuit_family`.
        template: If set, one of "bell", "ghz", "qft", "variational" --
            dispatches to that family's dedicated builder instead of the
            generic layer-based one. `None` means "generic, styled by the
            fields below."
        depth_fraction_range: (low, high) fraction of the requested
            [min_depth, max_depth] window this archetype samples its
            target depth (or padding budget, for templates) from.
        entangling_probability: Probability that a layer placement
            (given at least one other free qubit) is a multi-qubit
            entangling gate rather than a single-qubit gate. Ignored by
            templates' core structure; still used for any padding layers.
        parameterized_probability: Probability that a chosen single- or
            two-qubit gate is drawn from the parameterized pool rather
            than the fixed-gate pool.
        gate_density: Probability that a given free qubit receives a
            gate at all during a layer (the remainder stay idle that
            layer). Lower values produce sparser circuits.
        measurement_density: Fraction of qubits measured at the end of
            the circuit (rounded up, minimum of 1).
        weight: Relative frequency of this archetype in the generated
            dataset, used for weighted round-robin assignment.
        single_qubit_pool: Optional override for the fixed single-qubit
            gate pool (e.g. the Clifford subset). `None` uses the module
            default.
        two_qubit_pool: Optional override for the fixed two-qubit gate
            pool. `None` uses the module default.
        allow_three_qubit_gates: Whether this archetype may use
            three-qubit gates (currently CCX). Set to `False` for
            families whose gate set must stay closed under a specific
            property -- e.g. "random_clifford", since CCX (Toffoli) is
            not a Clifford gate and its inclusion would silently break
            that family's defining constraint.
    """

    name: str
    template: str | None = None
    depth_fraction_range: tuple[float, float] = (0.0, 1.0)
    entangling_probability: float = 0.35
    parameterized_probability: float = 0.3
    gate_density: float = 1.0
    measurement_density: float = 1.0
    weight: float = 1.0
    single_qubit_pool: tuple[str, ...] | None = None
    two_qubit_pool: tuple[str, ...] | None = None
    allow_three_qubit_gates: bool = True


# Original small/demo-scale archetype set (structural biases only, no
# templates). Kept as-is for `generate_circuits` backward compatibility.
DEFAULT_ARCHETYPES: tuple[CircuitArchetype, ...] = (
    CircuitArchetype(name="low_depth", depth_fraction_range=(0.0, 0.3)),
    CircuitArchetype(name="high_depth", depth_fraction_range=(0.7, 1.0)),
    CircuitArchetype(name="low_entanglement", entangling_probability=0.05),
    CircuitArchetype(name="heavy_entanglement", entangling_probability=0.75),
    CircuitArchetype(
        name="single_qubit_heavy", entangling_probability=0.05, parameterized_probability=0.5
    ),
    CircuitArchetype(name="multi_qubit_heavy", entangling_probability=0.85),
    CircuitArchetype(name="parameterized_heavy", parameterized_probability=0.85),
    CircuitArchetype(name="sparse", gate_density=0.4),
    CircuitArchetype(name="dense", gate_density=1.0, entangling_probability=0.5),
    CircuitArchetype(name="partial_measurement", measurement_density=0.5),
)


def _build_default_families(
    base_entangling_probability: float, base_parameterized_probability: float
) -> tuple[CircuitArchetype, ...]:
    """Build the 13 research-dataset families, threading the configurable
    baseline entangling/parameterized probabilities through as the
    reference point each family biases relative to.

    Families cover: two canonical templates (Bell, GHZ), one deterministic
    algorithmic template (QFT), one gate-pool-restricted family (random
    Clifford), one explicit ansatz template (variational), and eight
    structural-style archetypes spanning depth, entanglement, sparsity,
    rotation density, and measurement density. All default to equal
    weight, i.e. an approximately balanced dataset.
    """
    clip = lambda value: min(1.0, max(0.0, value))

    return (
        CircuitArchetype(name="bell_state", template="bell"),
        CircuitArchetype(name="ghz", template="ghz"),
        CircuitArchetype(name="qft_inspired", template="qft"),
        CircuitArchetype(
            name="random_clifford",
            entangling_probability=clip(base_entangling_probability * 1.1),
            parameterized_probability=0.0,
            single_qubit_pool=_CLIFFORD_SINGLE_QUBIT_GATES,
            two_qubit_pool=_CLIFFORD_TWO_QUBIT_GATES,
            allow_three_qubit_gates=False,
        ),
        CircuitArchetype(
            name="universal_random",
            entangling_probability=base_entangling_probability,
            parameterized_probability=base_parameterized_probability,
        ),
        CircuitArchetype(
            name="variational",
            template="variational",
            parameterized_probability=clip(base_parameterized_probability * 2.5),
        ),
        CircuitArchetype(
            name="entanglement_heavy", entangling_probability=clip(base_entangling_probability * 2.2)
        ),
        CircuitArchetype(name="sparse", gate_density=0.35),
        CircuitArchetype(
            name="dense", gate_density=1.0, entangling_probability=clip(base_entangling_probability * 1.4)
        ),
        CircuitArchetype(name="low_depth", depth_fraction_range=(0.0, 0.25)),
        CircuitArchetype(name="high_depth", depth_fraction_range=(0.75, 1.0)),
        CircuitArchetype(
            name="rotation_heavy",
            parameterized_probability=clip(base_parameterized_probability * 2.8),
            entangling_probability=clip(base_entangling_probability * 0.3),
        ),
        CircuitArchetype(name="measurement_heavy", measurement_density=1.0),
    )


def _weighted_round_robin_schedule(
    archetypes: Sequence[CircuitArchetype], number_of_circuits: int
) -> list[CircuitArchetype]:
    """Build an assignment of archetypes to circuit slots.

    Uses weighted round-robin (repeatedly cycling through archetypes,
    each appearing proportionally to its `weight`) rather than random
    sampling, so every archetype is guaranteed representation even for
    small `number_of_circuits` -- random sampling could, by chance, omit
    an archetype entirely from a small dataset or a single batch.

    Args:
        archetypes: Archetypes to schedule.
        number_of_circuits: Total number of slots to fill.

    Returns:
        A list of length `number_of_circuits`, each entry an archetype.
    """
    total_weight = sum(a.weight for a in archetypes)
    schedule: list[CircuitArchetype] = []

    remaining = number_of_circuits
    shares: list[tuple[CircuitArchetype, int]] = []
    for archetype in archetypes:
        share = int(number_of_circuits * (archetype.weight / total_weight))
        shares.append((archetype, share))
        remaining -= share

    for i in range(remaining):
        archetype, share = shares[i % len(shares)]
        shares[i % len(shares)] = (archetype, share + 1)

    per_archetype_lists = [[a] * s for a, s in shares]
    index = 0
    while len(schedule) < number_of_circuits:
        bucket = per_archetype_lists[index % len(per_archetype_lists)]
        if bucket:
            schedule.append(bucket.pop())
        index += 1
        if index > number_of_circuits * len(archetypes) + 1:
            break  # safety valve; should be unreachable given the share math above

    return schedule[:number_of_circuits]


def _sample_target_depth(
    archetype: CircuitArchetype, min_depth: int, max_depth: int, rng: random.Random
) -> int:
    """Sample a target layer count from an archetype's depth fraction range."""
    low_fraction, high_fraction = archetype.depth_fraction_range
    span = max_depth - min_depth
    low = min_depth + span * low_fraction
    high = min_depth + span * high_fraction
    return max(1, round(rng.uniform(low, high)))


def _build_layer(
    qc: QuantumCircuit,
    archetype: CircuitArchetype,
    rng: random.Random,
) -> None:
    """Add one layer of gates to `qc`, each qubit used at most once.

    Iterates over qubits in random order, greedily assigning each
    still-free qubit a gate (or leaving it idle, per
    `archetype.gate_density`) and consuming any other qubits that gate
    also touches, so gates within a layer never share a qubit -- this is
    what lets the emergent circuit depth track the requested target.
    """
    free_qubits = list(range(qc.num_qubits))
    rng.shuffle(free_qubits)

    while free_qubits:
        qubit = free_qubits.pop()

        if rng.random() > archetype.gate_density:
            continue  # qubit stays idle this layer

        can_entangle = len(free_qubits) >= 1
        if can_entangle and rng.random() < archetype.entangling_probability:
            _place_entangling_gate(qc, qubit, free_qubits, archetype, rng)
        else:
            _place_single_qubit_gate(qc, qubit, archetype, rng)


def _place_single_qubit_gate(
    qc: QuantumCircuit, qubit: int, archetype: CircuitArchetype, rng: random.Random
) -> None:
    """Apply one single-qubit gate (fixed or parameterized) to `qubit`."""
    if rng.random() < archetype.parameterized_probability:
        gate_name = rng.choice(_PARAMETERIZED_SINGLE_QUBIT_GATES)
        angle = rng.uniform(*_ANGLE_RANGE)
        getattr(qc, gate_name)(angle, qubit)
    else:
        pool = archetype.single_qubit_pool or _FIXED_SINGLE_QUBIT_GATES
        gate_name = rng.choice(pool)
        getattr(qc, gate_name)(qubit)


def _place_entangling_gate(
    qc: QuantumCircuit,
    qubit: int,
    free_qubits: list[int],
    archetype: CircuitArchetype,
    rng: random.Random,
) -> None:
    """Apply a multi-qubit entangling gate involving `qubit` and consume partners.

    Mutates `free_qubits` in place to remove whichever additional
    qubit(s) the chosen gate consumes, so the calling layer loop doesn't
    reuse them.
    """
    use_three_qubit = (
        archetype.allow_three_qubit_gates
        and len(free_qubits) >= 2
        and rng.random() < _THREE_QUBIT_GATE_PROBABILITY
    )

    if use_three_qubit:
        partner_a = free_qubits.pop(rng.randrange(len(free_qubits)))
        partner_b = free_qubits.pop(rng.randrange(len(free_qubits)))
        gate_name = rng.choice(_THREE_QUBIT_GATES)
        getattr(qc, gate_name)(qubit, partner_a, partner_b)
        return

    partner = free_qubits.pop(rng.randrange(len(free_qubits)))
    if rng.random() < archetype.parameterized_probability:
        gate_name = rng.choice(_PARAMETERIZED_TWO_QUBIT_GATES)
        angle = rng.uniform(*_ANGLE_RANGE)
        getattr(qc, gate_name)(angle, qubit, partner)
    else:
        pool = archetype.two_qubit_pool or _FIXED_TWO_QUBIT_GATES
        gate_name = rng.choice(pool)
        getattr(qc, gate_name)(qubit, partner)


def _build_generic_layers(
    qc: QuantumCircuit,
    archetype: CircuitArchetype,
    target_depth: int,
    max_gates: int,
    rng: random.Random,
) -> None:
    """Add up to `target_depth` generic layers, stopping early at `max_gates`."""
    for _ in range(target_depth):
        if qc.size() >= max_gates:
            break
        _build_layer(qc, archetype, rng)


def _pad_to_target_depth(
    qc: QuantumCircuit,
    archetype: CircuitArchetype,
    target_depth: int,
    max_gates: int,
    rng: random.Random,
) -> None:
    """Add generic layers on top of a template circuit to reach `target_depth`.

    Used after building a fixed template (Bell/GHZ/QFT) so the emergent
    depth still varies across the requested [min_depth, max_depth] window
    like the other families, without disturbing the template's
    recognizable core structure (padding is appended after it).
    """
    current_depth = qc.depth()
    layers_to_add = max(0, target_depth - current_depth)
    for _ in range(layers_to_add):
        if qc.size() >= max_gates:
            break
        _build_layer(qc, archetype, rng)


def _build_bell_template(qc: QuantumCircuit) -> None:
    """Build a Bell-state motif: H + CX on each disjoint qubit pair.

    Generalizes the canonical 2-qubit Bell state to arbitrary register
    width by tiling independent Bell pairs across the qubits. An odd
    leftover qubit gets a lone Hadamard (no partner to entangle with).
    """
    num_qubits = qc.num_qubits
    for i in range(0, num_qubits - 1, 2):
        qc.h(i)
        qc.cx(i, i + 1)
    if num_qubits % 2 == 1:
        qc.h(num_qubits - 1)


def _build_ghz_template(qc: QuantumCircuit) -> None:
    """Build a GHZ state: H on qubit 0, then a CX chain across all qubits."""
    qc.h(0)
    for i in range(qc.num_qubits - 1):
        qc.cx(i, i + 1)


def _build_qft_template(qc: QuantumCircuit) -> None:
    """Build a standard Quantum Fourier Transform circuit.

    For each qubit i: Hadamard, then controlled-phase rotations from
    every later qubit j with angle pi / 2^(j - i) -- the standard QFT
    construction -- followed by a final swap layer to reverse qubit
    order. Deterministic given `num_qubits`; diversity across the depth
    range comes from the padding layer appended afterward.
    """
    num_qubits = qc.num_qubits
    for i in range(num_qubits):
        qc.h(i)
        for j in range(i + 1, num_qubits):
            angle = math.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
    for i in range(num_qubits // 2):
        qc.swap(i, num_qubits - 1 - i)


def _build_variational_template(
    qc: QuantumCircuit, archetype: CircuitArchetype, target_depth: int, max_gates: int, rng: random.Random
) -> None:
    """Build a hardware-efficient variational ansatz.

    Repeats a block of [RY+RZ rotation on every qubit] followed by [a
    ring of CX entanglers] -- the standard "hardware-efficient ansatz"
    shape used in variational quantum algorithms (VQE/QAOA-style
    circuits) -- for a number of repetitions derived from the sampled
    target depth (roughly `_VARIATIONAL_LAYERS_PER_BLOCK` circuit layers
    per block), capped by `max_gates`.
    """
    num_blocks = max(1, target_depth // _VARIATIONAL_LAYERS_PER_BLOCK)
    num_qubits = qc.num_qubits

    for _ in range(num_blocks):
        if qc.size() >= max_gates:
            break
        for q in range(num_qubits):
            qc.ry(rng.uniform(*_ANGLE_RANGE), q)
            qc.rz(rng.uniform(*_ANGLE_RANGE), q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        if num_qubits > 2:
            qc.cx(num_qubits - 1, 0)


def _add_measurements(
    qc: QuantumCircuit, archetype: CircuitArchetype, rng: random.Random
) -> None:
    """Add a final measurement layer, measuring a subset of qubits.

    Measurements are added once, after all gate layers, so they appear
    near the end rather than being scattered through the circuit.
    `archetype.measurement_density` controls what fraction of qubits are
    measured (at least one).
    """
    num_to_measure = max(1, round(qc.num_qubits * archetype.measurement_density))
    measured_qubits = sorted(rng.sample(range(qc.num_qubits), num_to_measure))

    creg_size = len(measured_qubits)
    clbits = list(range(creg_size))
    if qc.num_clbits < creg_size:
        # Explicit, fixed register name rather than Qiskit's default
        # auto-incrementing name (e.g. "c0", "c1", ...): the default
        # counter is global to the Qiskit process, so its value depends
        # on how many registers have been created elsewhere in the run --
        # not on this module's own RNG -- which would silently break
        # exact QASM-text reproducibility for a given random_seed.
        qc.add_register(ClassicalRegister(creg_size, name="c"))

    qc.measure(measured_qubits, clbits)


def _generate_single_circuit(
    name: str,
    archetype: CircuitArchetype,
    min_qubits: int,
    max_qubits: int,
    min_depth: int,
    max_depth: int,
    max_gates: int,
    rng: random.Random,
) -> QuantumCircuit:
    """Build one QuantumCircuit for the given archetype and RNG.

    Dispatches to a named template builder if `archetype.template` is
    set, then (for templates) pads with generic layers to vary depth
    across the requested range; otherwise builds entirely from generic
    layers. Measurements are always added last.
    """
    num_qubits = rng.randint(min_qubits, max_qubits)
    target_depth = _sample_target_depth(archetype, min_depth, max_depth, rng)

    qc = QuantumCircuit(num_qubits, name=name)

    if archetype.template == "bell":
        _build_bell_template(qc)
        _pad_to_target_depth(qc, archetype, target_depth, max_gates, rng)
    elif archetype.template == "ghz":
        _build_ghz_template(qc)
        _pad_to_target_depth(qc, archetype, target_depth, max_gates, rng)
    elif archetype.template == "qft":
        _build_qft_template(qc)
        _pad_to_target_depth(qc, archetype, target_depth, max_gates, rng)
    elif archetype.template == "variational":
        _build_variational_template(qc, archetype, target_depth, max_gates, rng)
    else:
        _build_generic_layers(qc, archetype, target_depth, max_gates, rng)

    _add_measurements(qc, archetype, rng)
    return qc


def _canonical_signature(analysis: dict[str, Any]) -> str:
    """Build a coarse, canonical structural signature string for a circuit.

    Deliberately coarse (num_qubits, depth, sorted gate counts) rather
    than a full canonical/isomorphism-based comparison -- cheap to
    compute (reuses `analyzer.analyze_circuit`) and effective enough to
    catch near-identical circuits. It will not catch circuits that are
    structurally distinct but statistically similar, nor circuits
    isomorphic under qubit relabeling with different gate orders; a
    future version could upgrade this to a true canonical-form hash if
    stricter deduplication becomes necessary.
    """
    gate_repr = ",".join(f"{name}:{count}" for name, count in sorted(analysis["gate_counts"].items()))
    return f"{analysis['num_qubits']}|{analysis['depth']}|{gate_repr}"


def _signature_hash(signature: str) -> str:
    """Hash a canonical signature to a fixed-size hex digest.

    Storing only the hash (not the variable-length signature string)
    keeps the duplicate-tracking set's memory footprint bounded and
    predictable even across tens of thousands of circuits and many
    batches -- important since this set persists for the whole
    multi-batch run, not just one batch.
    """
    return hashlib.blake2b(signature.encode("utf-8"), digest_size=16).hexdigest()


def _generate_unique_circuit(
    name: str,
    archetype: CircuitArchetype,
    min_qubits: int,
    max_qubits: int,
    min_depth: int,
    max_depth: int,
    max_gates: int,
    rng: random.Random,
    seen_hashes: set[str],
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Generate a circuit, retrying on structural-signature collision.

    Computes `analyzer.analyze_circuit` once per attempt and reuses that
    same analysis for both the duplicate check and (by the caller) the
    metadata row -- avoiding a second, redundant analysis pass.

    Returns:
        A tuple of (circuit, its analysis dict).
    """
    qc = _generate_single_circuit(name, archetype, min_qubits, max_qubits, min_depth, max_depth, max_gates, rng)
    analysis = analyze_circuit(qc)
    signature_hash = _signature_hash(_canonical_signature(analysis))

    attempts = 0
    while signature_hash in seen_hashes and attempts < _MAX_DUPLICATE_RETRIES:
        qc = _generate_single_circuit(
            name, archetype, min_qubits, max_qubits, min_depth, max_depth, max_gates, rng
        )
        analysis = analyze_circuit(qc)
        signature_hash = _signature_hash(_canonical_signature(analysis))
        attempts += 1

    seen_hashes.add(signature_hash)
    return qc, analysis


def _write_qasm_file(qc: QuantumCircuit, path: Path) -> None:
    """Serialize a circuit to OpenQASM 3 and write it to disk.

    Raises:
        GenerationError: If the circuit cannot be exported or written.
    """
    try:
        qasm_text = qasm3_dumps(qc)
    except Exception as exc:
        raise GenerationError(f"Failed to export circuit '{qc.name}' to OpenQASM 3: {exc}") from exc

    try:
        path.write_text(qasm_text)
    except OSError as exc:
        raise GenerationError(f"Failed to write '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# Small/demo-scale API (backward compatible)
# ---------------------------------------------------------------------------


def _build_metadata_row(filename: str, archetype_name: str, qc: QuantumCircuit) -> dict[str, Any]:
    """Build one metadata.csv row, reusing analyzer/feature_extractor logic."""
    analysis = analyze_circuit(qc)
    features = extract_features(qc)

    return {
        "filename": filename,
        "archetype": archetype_name,
        "num_qubits": analysis["num_qubits"],
        "num_clbits": analysis["num_clbits"],
        "depth": analysis["depth"],
        "total_operations": analysis["total_operations"],
        "single_qubit_gates": features["single_qubit_gates"],
        "two_qubit_gates": features["two_qubit_gates"],
        "three_qubit_gates": features["three_qubit_gates"],
        "measurement_gates": features["measurement_gates"],
        "parameterized_gates": features["parameterized_gates"],
        "entangling_gates": features["entangling_gates"],
        "gate_distribution": json.dumps(analysis["gate_counts"], sort_keys=True),
    }


def _write_metadata_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write all metadata rows to a single CSV file."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    try:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise GenerationError(f"Failed to write metadata file '{path}': {exc}") from exc


def generate_circuits(
    output_directory: str | Path = "generated/",
    number_of_circuits: int = 1000,
    min_qubits: int = 2,
    max_qubits: int = 8,
    min_depth: int = 3,
    max_depth: int = 40,
    random_seed: int | None = None,
    archetypes: Sequence[CircuitArchetype] | None = None,
) -> list[QuantumCircuit]:
    """Generate a diverse collection of circuits, exported as .qasm + metadata.

    This is the original small/demo-scale API: it keeps every generated
    `QuantumCircuit` in memory and returns them as a list, which is fine
    at hundreds-to-low-thousands of circuits but is NOT used by
    `generate_dataset_batches` (see that function for the memory-bounded,
    large-scale equivalent).

    Args:
        output_directory: Directory to write `circuit_NNNN.qasm` files
            and `metadata.csv` into. Created if it doesn't exist.
        number_of_circuits: Total number of circuits to generate.
        min_qubits: Minimum number of qubits per circuit (inclusive).
        max_qubits: Maximum number of qubits per circuit (inclusive).
        min_depth: Minimum target circuit depth (inclusive; the realized
            depth is an emergent property, see module docstring).
        max_depth: Maximum target circuit depth (inclusive).
        random_seed: Optional seed for full reproducibility. `None` uses
            nondeterministic system randomness.
        archetypes: Optional custom archetype set, overriding
            `DEFAULT_ARCHETYPES`.

    Returns:
        A list of the generated `QuantumCircuit` objects, in generation
        order.

    Raises:
        ValueError: If any of the numeric range arguments are invalid.
        GenerationError: If circuit export or metadata writing fails.
    """
    if min_qubits < 1 or max_qubits < min_qubits:
        raise ValueError(f"Invalid qubit range: min_qubits={min_qubits}, max_qubits={max_qubits}")
    if min_depth < 1 or max_depth < min_depth:
        raise ValueError(f"Invalid depth range: min_depth={min_depth}, max_depth={max_depth}")
    if number_of_circuits < 1:
        raise ValueError(f"number_of_circuits must be positive, got {number_of_circuits}")

    active_archetypes = archetypes or DEFAULT_ARCHETYPES
    output_path = Path(output_directory)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GenerationError(f"Failed to create output directory '{output_path}': {exc}") from exc

    master_rng = random.Random(random_seed)
    schedule = _weighted_round_robin_schedule(active_archetypes, number_of_circuits)

    seen_hashes: set[str] = set()
    circuits: list[QuantumCircuit] = []
    metadata_rows: list[dict[str, Any]] = []

    # No max_gates ceiling in this legacy API; use a large sentinel so
    # behavior is unchanged from before max_gates was introduced.
    unlimited_gates = 10**9

    for index, archetype in enumerate(schedule, start=1):
        filename = f"circuit_{index:04d}.qasm"
        circuit_name = f"circuit_{index:04d}"
        circuit_seed = master_rng.randrange(2**32)
        circuit_rng = random.Random(circuit_seed)

        qc, _ = _generate_unique_circuit(
            circuit_name, archetype, min_qubits, max_qubits, min_depth, max_depth,
            unlimited_gates, circuit_rng, seen_hashes,
        )
        circuits.append(qc)

        _write_qasm_file(qc, output_path / filename)
        metadata_rows.append(_build_metadata_row(filename, archetype.name, qc))

    _write_metadata_csv(metadata_rows, output_path / "metadata.csv")
    return circuits


def _summarize_demo(circuits: list[QuantumCircuit]) -> None:
    """Print summary statistics for a demonstration batch of circuits."""
    analyses = [analyze_circuit(qc) for qc in circuits]
    features = [extract_features(qc) for qc in circuits]

    depths = [a["depth"] for a in analyses]
    qubit_counts = [a["num_qubits"] for a in analyses]
    gate_counts = [a["total_operations"] for a in analyses]

    gate_type_totals: dict[str, int] = {}
    for a in analyses:
        for gate_name, count in a["gate_counts"].items():
            gate_type_totals[gate_name] = gate_type_totals.get(gate_name, 0) + count

    num_parameterized = sum(1 for f in features if f["parameterized_gates"] > 0)
    num_entangling_heavy = sum(1 for f in features if f["entangling_gates"] >= f["total_operations"] * 0.3)

    print("=" * 40)
    print("Circuit Generation Demo Summary")
    print("=" * 40)
    print(f"Circuits generated       : {len(circuits)}")
    print(f"Average depth            : {sum(depths) / len(depths):.2f}")
    print(f"Average qubits           : {sum(qubit_counts) / len(qubit_counts):.2f}")
    print(f"Average gate count       : {sum(gate_counts) / len(gate_counts):.2f}")
    print(f"Parameterized circuits   : {num_parameterized}")
    print(f"Entangling-heavy circuits: {num_entangling_heavy}")
    print()
    print("Gate type distribution (totals across all circuits):")
    for gate_name, count in sorted(gate_type_totals.items()):
        print(f"  {gate_name:<10}: {count}")


# ---------------------------------------------------------------------------
# Large-scale, batched, memory-bounded API
# ---------------------------------------------------------------------------


@dataclass
class _RunningStats:
    """Scalar running totals for one batch. No per-circuit records retained."""

    count: int = 0
    depth_sum: int = 0
    qubit_sum: int = 0
    gate_count_sum: int = 0
    gate_type_totals: dict[str, int] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)

    def update(self, analysis: dict[str, Any], family_name: str) -> None:
        self.count += 1
        self.depth_sum += analysis["depth"]
        self.qubit_sum += analysis["num_qubits"]
        self.gate_count_sum += analysis["total_operations"]
        for gate_name, gate_count in analysis["gate_counts"].items():
            self.gate_type_totals[gate_name] = self.gate_type_totals.get(gate_name, 0) + gate_count
        self.family_counts[family_name] = self.family_counts.get(family_name, 0) + 1

    @property
    def average_depth(self) -> float:
        return self.depth_sum / self.count if self.count else 0.0

    @property
    def average_qubits(self) -> float:
        return self.qubit_sum / self.count if self.count else 0.0

    @property
    def average_gate_count(self) -> float:
        return self.gate_count_sum / self.count if self.count else 0.0


@dataclass
class BatchSummary:
    """Summary of one completed batch. Does not retain any circuit objects."""

    batch_number: int
    batch_directory: Path
    circuit_count: int
    generation_seconds: float


@dataclass
class GenerationSummary:
    """Summary of a full (possibly multi-batch) generation run."""

    output_directory: Path
    total_circuits: int
    num_batches: int
    batches: list[BatchSummary]


_BATCH_METADATA_FIELDNAMES: tuple[str, ...] = (
    "filename",
    "number_of_qubits",
    "depth",
    "gate_count",
    "gate_distribution",
    "circuit_family",
    "generation_seed",
)


def _print_batch_statistics(batch_number: int, stats: _RunningStats, elapsed_seconds: float) -> None:
    """Print the required post-batch statistics.

    This is one of two places in the module that print (the other is the
    `__main__` demo block) -- an intentional, documented exception to
    "modules don't print" for this specific large-scale entry point,
    since progress visibility genuinely matters at 50,000+ circuit scale
    and there is no separate orchestration layer above this function for
    batch generation the way `main.py` sits above the analysis pipeline.
    """
    print("=" * 50)
    print(f"Batch {batch_number:04d} complete -- {stats.count} circuits")
    print("=" * 50)
    print(f"Average depth      : {stats.average_depth:.2f}")
    print(f"Average qubits     : {stats.average_qubits:.2f}")
    print(f"Average gate count : {stats.average_gate_count:.2f}")
    print(f"Generation time    : {elapsed_seconds:.2f}s")
    print()
    print("Gate distribution:")
    for gate_name, count in sorted(stats.gate_type_totals.items()):
        print(f"  {gate_name:<10}: {count}")
    print()
    print("Family distribution:")
    for family_name, count in sorted(stats.family_counts.items()):
        print(f"  {family_name:<18}: {count}")
    print()


def _generate_batch(
    batch_dir: Path,
    schedule: list[CircuitArchetype],
    starting_index: int,
    min_qubits: int,
    max_qubits: int,
    min_depth: int,
    max_depth: int,
    max_gates: int,
    master_rng: random.Random,
    seen_hashes: set[str],
) -> _RunningStats:
    """Generate, write, and release every circuit in one batch.

    Only one `QuantumCircuit` is alive at a time: it's built, analyzed,
    written to disk, folded into `stats`, and then goes out of scope
    (nothing retains a reference to it) before the next one is built.
    Metadata rows are streamed to `metadata.csv` as each circuit is
    processed, rather than accumulated in memory.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    stats = _RunningStats()

    metadata_path = batch_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BATCH_METADATA_FIELDNAMES)
        writer.writeheader()

        for offset, archetype in enumerate(schedule):
            index = starting_index + offset
            filename = f"circuit_{index:06d}.qasm"
            circuit_name = f"circuit_{index:06d}"
            circuit_seed = master_rng.randrange(2**32)
            circuit_rng = random.Random(circuit_seed)

            qc, analysis = _generate_unique_circuit(
                circuit_name, archetype, min_qubits, max_qubits, min_depth, max_depth,
                max_gates, circuit_rng, seen_hashes,
            )

            _write_qasm_file(qc, batch_dir / filename)

            writer.writerow(
                {
                    "filename": filename,
                    "number_of_qubits": analysis["num_qubits"],
                    "depth": analysis["depth"],
                    "gate_count": analysis["total_operations"],
                    "gate_distribution": json.dumps(analysis["gate_counts"], sort_keys=True),
                    "circuit_family": archetype.name,
                    "generation_seed": circuit_seed,
                }
            )

            stats.update(analysis, archetype.name)
            # `qc` goes out of scope at the next loop iteration; no list
            # or collection anywhere in this function retains it.

    return stats


def generate_dataset_batches(
    output_directory: str | Path = "generated",
    total_circuits: int = 50_000,
    batch_size: int = 1_000,
    min_qubits: int = 2,
    max_qubits: int = 12,
    min_depth: int = 3,
    max_depth: int = 60,
    max_gates: int = 200,
    entangling_gate_probability: float = 0.35,
    parameterized_gate_probability: float = 0.3,
    random_seed: int | None = None,
    families: Sequence[CircuitArchetype] | None = None,
) -> GenerationSummary:
    """Generate a large, batched, statistically balanced circuit dataset.

    Writes `output_directory/batch_0001/`, `batch_0002/`, ... -- each
    containing its own `.qasm` files and `metadata.csv` -- so any single
    batch folder is, on its own, exactly the flat-directory shape
    `dataset_generator.generate_dataset` already expects. Point it at one
    batch folder at a time; no changes to that module are needed.

    Only one `QuantumCircuit` is held in memory at a time (see
    `_generate_batch`); this function is safe to run at 50,000+ circuits.

    Args:
        output_directory: Root directory for `batch_NNNN/` subfolders.
        total_circuits: Total number of circuits across all batches.
        batch_size: Circuits per batch (the last batch may be smaller).
        min_qubits: Minimum number of qubits per circuit (inclusive).
        max_qubits: Maximum number of qubits per circuit (inclusive).
        min_depth: Minimum target circuit depth (inclusive).
        max_depth: Maximum target circuit depth (inclusive).
        max_gates: Hard ceiling on gate count per circuit (excluding the
            final measurement layer), regardless of target depth.
        entangling_gate_probability: Baseline entangling-gate probability
            used to construct the default family set (families bias
            relative to this baseline; ignored if `families` is given).
        parameterized_gate_probability: Baseline parameterized-gate
            probability used to construct the default family set (same
            caveat as above).
        random_seed: Optional seed for full reproducibility of the whole
            multi-batch run.
        families: Optional custom family set, overriding the built-in 13
            families (Bell, GHZ, QFT, random Clifford, universal random,
            variational, entanglement-heavy, sparse, dense, low-depth,
            high-depth, rotation-heavy, measurement-heavy).

    Returns:
        A `GenerationSummary` with per-batch metadata. Does NOT include
        any `QuantumCircuit` objects or per-circuit records -- read the
        written `.qasm` files and `metadata.csv` files for that.

    Raises:
        ValueError: If any of the numeric range arguments are invalid.
        GenerationError: If circuit export or metadata writing fails.
    """
    if min_qubits < 1 or max_qubits < min_qubits:
        raise ValueError(f"Invalid qubit range: min_qubits={min_qubits}, max_qubits={max_qubits}")
    if min_depth < 1 or max_depth < min_depth:
        raise ValueError(f"Invalid depth range: min_depth={min_depth}, max_depth={max_depth}")
    if max_gates < 1:
        raise ValueError(f"max_gates must be positive, got {max_gates}")
    if total_circuits < 1:
        raise ValueError(f"total_circuits must be positive, got {total_circuits}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    active_families = families or _build_default_families(
        entangling_gate_probability, parameterized_gate_probability
    )

    output_path = Path(output_directory)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GenerationError(f"Failed to create output directory '{output_path}': {exc}") from exc

    master_rng = random.Random(random_seed)
    seen_hashes: set[str] = set()

    num_batches = math.ceil(total_circuits / batch_size)
    overall_index = 0
    batch_summaries: list[BatchSummary] = []

    for batch_number in range(1, num_batches + 1):
        circuits_remaining = total_circuits - (batch_number - 1) * batch_size
        batch_circuit_count = min(batch_size, circuits_remaining)
        batch_dir = output_path / f"batch_{batch_number:04d}"
        schedule = _weighted_round_robin_schedule(active_families, batch_circuit_count)

        start_time = time.monotonic()
        stats = _generate_batch(
            batch_dir, schedule, overall_index + 1,
            min_qubits, max_qubits, min_depth, max_depth, max_gates,
            master_rng, seen_hashes,
        )
        elapsed_seconds = time.monotonic() - start_time

        overall_index += batch_circuit_count
        _print_batch_statistics(batch_number, stats, elapsed_seconds)

        batch_summaries.append(
            BatchSummary(
                batch_number=batch_number,
                batch_directory=batch_dir,
                circuit_count=batch_circuit_count,
                generation_seconds=elapsed_seconds,
            )
        )

    return GenerationSummary(
        output_directory=output_path,
        total_circuits=overall_index,
        num_batches=num_batches,
        batches=batch_summaries,
    )


if __name__ == "__main__":
    # Demonstration only: 2 batches of 100 circuits each. The engine
    # itself supports arbitrary scale (see generate_dataset_batches'
    # docstring) -- this is intentionally NOT a 50,000-circuit run.
    summary = generate_dataset_batches(
        output_directory="generated_batches",
        total_circuits=200,
        batch_size=100,
        min_qubits=2,
        max_qubits=8,
        min_depth=3,
        max_depth=40,
        max_gates=200,
        random_seed=123,
    )
    print(f"Run complete: {summary.total_circuits} circuits across {summary.num_batches} batches")
    for batch in summary.batches:
        print(f"  {batch.batch_directory}: {batch.circuit_count} circuits in {batch.generation_seconds:.2f}s")
