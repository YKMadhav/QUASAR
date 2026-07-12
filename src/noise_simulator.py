"""
noise_simulator.py
-------------------
Single Responsibility:
    Given a QuantumCircuit, run an ideal (noiseless) simulation and a
    noisy simulation, then compute reliability metrics that compare them.

This module intentionally does NOT:
    - Extract structural or ML features (see analyzer.py, feature_extractor.py)
    - Build a training dataset (see future dataset_generator.py)
    - Train or run any machine learning model

Where this fits in the pipeline:
    Parser -> Analyzer -> Feature Extractor -> Noise Simulator
    -> Dataset Generator (future) -> Machine Learning (future)

Why this module must exist before dataset generation:
    `feature_extractor.py` produces the ML inputs ("X") -- structural
    facts about a circuit. It has no way to know how reliably that
    circuit actually runs; that information only exists once the circuit
    is executed under a noise model and compared to the ideal case. This
    module is the only place in the project that produces the ML targets
    ("y") -- estimated fidelity / noise impact -- that a future
    `dataset_generator.py` will pair with each circuit's feature dict.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

DEFAULT_SHOTS = 1024

# Thresholds for classifying Total Variation Distance (the module's
# "noise impact" metric -- see NOTE in `simulate_noise`'s docstring for
# why TVD was chosen over `1 - estimated_fidelity`) into a human-readable
# noise assessment. These are heuristic starting points appropriate for a
# v0.3 research prototype: they encode "what fraction of measurement
# outcomes were displaced by noise", not a hardware-validated tolerance.
# They should be recalibrated once real device data is available.
NOISE_ASSESSMENT_THRESHOLDS: dict[str, float] = {
    "LOW": 0.05,
    "MEDIUM": 0.15,
}

# Thresholds for classifying estimated (Bhattacharyya-based) fidelity into
# an overall circuit reliability rating. Same caveat as above: heuristic,
# not yet hardware-calibrated.
RELIABILITY_THRESHOLDS: dict[str, float] = {
    "HIGH": 0.95,
    "MEDIUM": 0.85,
}

# z-score for a 95% confidence margin on a binomial proportion.
_CONFIDENCE_Z_SCORE = 1.96


class CircuitSimulationError(Exception):
    """Raised when a circuit cannot be simulated (e.g. it has no
    measurement instructions, or the backend rejects it after transpile).

    Kept as a project-specific exception -- mirroring `QasmParsingError`
    in parser.py -- so callers can catch one stable error type regardless
    of which underlying Aer/Qiskit exception was actually raised.
    """


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for a synthetic, hardware-agnostic noise model.

    This is intentionally a plain data container: swapping to a
    hardware-calibrated noise model later does NOT require touching this
    class or its consumers -- see `build_noise_model`'s docstring for how
    a real backend's `NoiseModel` can be substituted entirely.

    Attributes:
        single_qubit_depolarizing_prob: Depolarizing error probability
            applied to single-qubit gates.
        two_qubit_depolarizing_prob: Depolarizing error probability
            applied to two-qubit gates.
        readout_error_prob: Symmetric probability of a bit-flip during
            measurement readout (P(read 1 | actual 0) == P(read 0 | actual 1)).
        single_qubit_gates: Gate names treated as single-qubit for the
            purpose of applying `single_qubit_depolarizing_prob`.
        two_qubit_gates: Gate names treated as two-qubit for the purpose
            of applying `two_qubit_depolarizing_prob`.
    """

    single_qubit_depolarizing_prob: float = 0.001
    two_qubit_depolarizing_prob: float = 0.01
    readout_error_prob: float = 0.02
    single_qubit_gates: tuple[str, ...] = (
        "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
        "sx", "rx", "ry", "rz", "u1", "u2", "u3",
    )
    two_qubit_gates: tuple[str, ...] = ("cx", "cz", "cp", "swap")


def _add_depolarizing_errors(noise_model: NoiseModel, config: NoiseConfig) -> None:
    """Add single- and two-qubit depolarizing errors to a noise model.

    Kept as its own function (rather than inline in `build_noise_model`)
    so a future noise channel -- e.g. thermal relaxation, amplitude
    damping -- can be added as a sibling function without editing this
    one, and so each channel can be unit-tested in isolation.
    """
    if config.single_qubit_depolarizing_prob > 0:
        single_qubit_error = depolarizing_error(
            config.single_qubit_depolarizing_prob, num_qubits=1
        )
        noise_model.add_all_qubit_quantum_error(
            single_qubit_error, list(config.single_qubit_gates)
        )

    if config.two_qubit_depolarizing_prob > 0:
        two_qubit_error = depolarizing_error(
            config.two_qubit_depolarizing_prob, num_qubits=2
        )
        noise_model.add_all_qubit_quantum_error(
            two_qubit_error, list(config.two_qubit_gates)
        )


def _add_readout_errors(noise_model: NoiseModel, config: NoiseConfig) -> None:
    """Add a symmetric readout (measurement) error to a noise model."""
    if config.readout_error_prob <= 0:
        return

    p_error = config.readout_error_prob
    readout_error = ReadoutError(
        [[1 - p_error, p_error], [p_error, 1 - p_error]]
    )
    noise_model.add_all_qubit_readout_error(readout_error)


# Ordered list of error-builder functions applied when constructing a
# default noise model. Each builder receives the (mutable) NoiseModel and
# the NoiseConfig, and adds its own error channel(s) to it. Extending the
# model with a new noise channel is a two-step change: write a new
# `_add_..._errors` function, then append it here.
_DEFAULT_ERROR_BUILDERS: tuple[Callable[[NoiseModel, NoiseConfig], None], ...] = (
    _add_depolarizing_errors,
    _add_readout_errors,
)


def build_noise_model(
    config: NoiseConfig | None = None,
    error_builders: Sequence[Callable[[NoiseModel, NoiseConfig], None]] | None = None,
) -> NoiseModel:
    """Build a synthetic NoiseModel from a NoiseConfig.

    Args:
        config: Noise parameters to use. Defaults to `NoiseConfig()`.
        error_builders: Ordered functions that each add one error channel
            to the model. Defaults to `_DEFAULT_ERROR_BUILDERS`. Exposed
            as a parameter so callers can add, remove, or reorder error
            channels without modifying this module.

    Returns:
        A configured `qiskit_aer.noise.NoiseModel`.

    Note on hardware-specific noise models:
        This function produces a *synthetic*, hardware-agnostic model
        suitable for early development and testing. To simulate a real
        device, construct a `NoiseModel` from that backend's calibration
        data (e.g. via `qiskit_aer.noise.NoiseModel.from_backend(backend)`
        against a real or fake IBM backend) and pass it directly to
        `simulate_noise(qc, noise_model=...)` instead of calling this
        function. `simulate_noise` never needs to know whether a model is
        synthetic or hardware-derived -- both are just `NoiseModel`
        instances.
    """
    active_config = config or NoiseConfig()
    active_builders = error_builders or _DEFAULT_ERROR_BUILDERS

    noise_model = NoiseModel()
    for add_errors in active_builders:
        add_errors(noise_model, active_config)

    return noise_model


def _run_simulation(
    qc: QuantumCircuit,
    shots: int,
    noise_model: NoiseModel | None,
) -> dict[str, int]:
    """Run a single simulation (ideal if `noise_model` is None, else noisy).

    Args:
        qc: The circuit to simulate. Must contain measurement instructions.
        shots: Number of simulation shots.
        noise_model: NoiseModel to apply, or None for an ideal simulation.

    Returns:
        A mapping of measured bitstring -> observed count.

    Raises:
        CircuitSimulationError: If the circuit cannot be transpiled or
            executed on the simulator backend.
    """
    backend = AerSimulator(noise_model=noise_model)

    try:
        transpiled = transpile(qc, backend)
        result = backend.run(transpiled, shots=shots).result()
        counts = result.get_counts()
    except Exception as exc:
        raise CircuitSimulationError(
            f"Failed to simulate circuit '{qc.name}': {exc}"
        ) from exc

    return dict(counts)


def _bhattacharyya_coefficient(
    ideal_counts: dict[str, int],
    noisy_counts: dict[str, int],
    shots: int,
) -> float:
    """Compute the Bhattacharyya coefficient (BC) between two distributions.

        BC = sum_x sqrt(p_ideal(x) * p_noisy(x))

    where the sum runs over every bitstring observed in either
    distribution and p(x) = count(x) / shots.

    BC is the shared building block for two of this module's metrics:
    `estimated_fidelity = BC ** 2` and `hellinger_distance = sqrt(1 - BC)`.
    It is computed once, here, so both derived metrics stay numerically
    consistent with each other and the outcome-distribution loop isn't
    duplicated.

    Args:
        ideal_counts: Bitstring -> count from the noiseless simulation.
        noisy_counts: Bitstring -> count from the noisy simulation.
        shots: Number of shots used for both simulations.

    Returns:
        A float in [0.0, 1.0].
    """
    all_outcomes = set(ideal_counts) | set(noisy_counts)

    bhattacharyya_coefficient = 0.0
    for outcome in all_outcomes:
        p_ideal = ideal_counts.get(outcome, 0) / shots
        p_noisy = noisy_counts.get(outcome, 0) / shots
        bhattacharyya_coefficient += math.sqrt(p_ideal * p_noisy)

    return min(max(bhattacharyya_coefficient, 0.0), 1.0)


def _estimate_fidelity(bhattacharyya_coefficient: float) -> float:
    """Estimate a classical fidelity from a precomputed Bhattacharyya coefficient.

    F = BC ** 2

    This is NOT the exact quantum state or process fidelity -- computing
    that would require the ideal statevector or full process tomography.
    It IS a standard, well-behaved classical similarity measure between
    two probability distributions built purely from measurement counts:
    it equals 1.0 when the distributions match exactly, trends toward 0.0
    as they diverge, and (unlike a single-bitstring success ratio) it
    naturally accounts for circuits with multiple valid ideal outcomes
    (e.g. a Bell state's "00"/"11" superposition).

    Note: because F is quadratic in BC near BC = 1, `1 - F` compresses
    small-to-moderate distributional differences (see `noise_impact` in
    `simulate_noise`'s docstring for why Total Variation Distance, not
    `1 - F`, is used as this module's primary noise-impact metric).

    Args:
        bhattacharyya_coefficient: Precomputed BC from `_bhattacharyya_coefficient`.

    Returns:
        A float in [0.0, 1.0].
    """
    return min(max(bhattacharyya_coefficient ** 2, 0.0), 1.0)


def _hellinger_distance(bhattacharyya_coefficient: float) -> float:
    """Compute the Hellinger distance from a precomputed Bhattacharyya coefficient.

        H = sqrt(1 - BC)

    A proper distance metric (symmetric, satisfies the triangle
    inequality) in [0.0, 1.0], derived at no extra computational cost
    from the same BC already used for `estimated_fidelity`. Included as
    a cross-check on the fidelity number using a differently-scaled
    (roughly linear rather than quadratic) view of the same distributional
    overlap.

    Args:
        bhattacharyya_coefficient: Precomputed BC from `_bhattacharyya_coefficient`.

    Returns:
        A float in [0.0, 1.0].
    """
    return math.sqrt(max(1.0 - bhattacharyya_coefficient, 0.0))


def _total_variation_distance(
    ideal_counts: dict[str, int],
    noisy_counts: dict[str, int],
    shots: int,
) -> float:
    """Compute the Total Variation Distance (TVD) between two distributions.

        TVD = (1/2) * sum_x |p_ideal(x) - p_noisy(x)|

    TVD is directly interpretable as "the fraction of total probability
    mass that must be moved to turn one distribution into the other."
    Unlike `1 - estimated_fidelity`, it scales roughly linearly (not
    quadratically) with small perturbations, which makes it more
    discriminating in the low-noise regime this project targets -- see
    `noise_impact` in `simulate_noise`'s docstring.

    KL divergence and Jensen-Shannon divergence were considered and
    intentionally not implemented for this v0.3 prototype: KL divergence
    is undefined/infinite whenever the noisy distribution has zero count
    for an outcome the ideal distribution produced (common with sparse,
    finite-shot count dictionaries), and JSD is log-based, more
    computationally involved, and largely redundant with the
    TVD/Hellinger pair already covering "linear-scale" and
    "square-root-scale" views of the same distributional difference.

    Args:
        ideal_counts: Bitstring -> count from the noiseless simulation.
        noisy_counts: Bitstring -> count from the noisy simulation.
        shots: Number of shots used for both simulations.

    Returns:
        A float in [0.0, 1.0].
    """
    all_outcomes = set(ideal_counts) | set(noisy_counts)

    total_absolute_difference = 0.0
    for outcome in all_outcomes:
        p_ideal = ideal_counts.get(outcome, 0) / shots
        p_noisy = noisy_counts.get(outcome, 0) / shots
        total_absolute_difference += abs(p_ideal - p_noisy)

    return min(max(total_absolute_difference / 2, 0.0), 1.0)


def _success_probability_margin_of_error(shots: int) -> float:
    """Return the 95% shot-noise margin of error for a binomial proportion.

    Uses p = 0.5 (the variance-maximizing case) since the true success
    probability isn't known a priori -- this gives the largest plausible
    shot-noise-only fluctuation band, i.e. a conservative (upper-bound)
    margin. Used to distinguish "the ideal vs. noisy success-probability
    gap reflects real noise" from "the gap is within expected sampling
    variation at this shot count" (see `simulate_noise`'s docstring).

    Args:
        shots: Number of shots used for the simulation.

    Returns:
        A float margin of error, e.g. 0.031 for shots=1024.
    """
    return _CONFIDENCE_Z_SCORE * math.sqrt(0.25 / shots)


def _classify_noise_assessment(total_variation_distance: float) -> str:
    """Classify a TVD value into a LOW/MEDIUM/HIGH noise assessment."""
    if total_variation_distance < NOISE_ASSESSMENT_THRESHOLDS["LOW"]:
        return "LOW"
    if total_variation_distance < NOISE_ASSESSMENT_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"


def _classify_circuit_reliability(estimated_fidelity: float) -> str:
    """Classify a fidelity value into a HIGH/MEDIUM/LOW reliability rating."""
    if estimated_fidelity >= RELIABILITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if estimated_fidelity >= RELIABILITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def _build_noise_assessment_explanation(
    total_variation_distance: float, noise_assessment: str
) -> str:
    """Generate a plain-language explanation of the noise assessment."""
    return (
        f"Noise assessment is {noise_assessment} because the total "
        f"variation distance between the ideal and noisy outcome "
        f"distributions is {total_variation_distance:.3f} "
        f"({total_variation_distance * 100:.1f}% of measurement "
        f"probability mass displaced by noise)."
    )


def _build_reliability_explanation(
    estimated_fidelity: float, circuit_reliability: str
) -> str:
    """Generate a plain-language explanation of the reliability rating."""
    return (
        f"Circuit reliability is rated {circuit_reliability} because the "
        f"estimated fidelity between ideal and noisy distributions is "
        f"{estimated_fidelity:.3f} ({estimated_fidelity * 100:.1f}%)."
    )


def _build_shot_noise_note(
    success_probability_gap: float, margin_of_error: float
) -> str:
    """Generate a note distinguishing real noise effects from shot noise."""
    if success_probability_gap <= margin_of_error:
        return (
            f"The {success_probability_gap:.3f} gap between ideal and "
            f"noisy success probability is within the expected shot-noise "
            f"margin of \u00b1{margin_of_error:.3f} at this shot count, and "
            f"should not, on its own, be read as evidence of noise-induced "
            f"degradation. The distribution-level metrics above (fidelity, "
            f"TVD, Hellinger distance) are more reliable indicators."
        )
    return (
        f"The {success_probability_gap:.3f} gap between ideal and noisy "
        f"success probability exceeds the expected shot-noise margin of "
        f"\u00b1{margin_of_error:.3f}, suggesting the difference reflects "
        f"real noise-induced degradation rather than sampling variation "
        f"alone."
    )


def simulate_noise(
    qc: QuantumCircuit,
    shots: int = DEFAULT_SHOTS,
    noise_model: NoiseModel | None = None,
) -> dict[str, Any]:
    """Run an ideal and a noisy simulation of a circuit and compare them.

    Args:
        qc: The circuit to simulate. Must contain measurement
            instructions (e.g. as produced by `parser.load_qasm_file`).
        shots: Number of shots to use for both simulations. Defaults to
            `DEFAULT_SHOTS` (1024).
        noise_model: The noise model to use for the noisy simulation.
            Defaults to `build_noise_model()` (a synthetic model). Pass a
            hardware-calibrated `NoiseModel` here to simulate a real
            device instead -- see `build_noise_model`'s docstring.

    Returns:
        A dictionary containing:
            - "total_shots" (int): Shots used for both simulations.
            - "ideal_counts" (dict[str, int]): Noiseless measurement counts.
            - "noisy_counts" (dict[str, int]): Noisy measurement counts.
            - "success_bitstring" (str): The most frequent outcome under
              the ideal simulation, used as the reference "success"
              outcome for the two probabilities below. Note: for circuits
              with multiple equally-valid ideal outcomes (e.g. Bell/GHZ
              states), this reference captures only one such outcome --
              `estimated_fidelity` and the distributional metrics below
              are the more robust indicators for such circuits since they
              consider the entire distribution.
            - "ideal_success_probability" (float): Fraction of ideal shots
              that produced `success_bitstring`.
            - "noisy_success_probability" (float): Fraction of noisy shots
              that produced `success_bitstring`.
            - "success_probability_margin_of_error" (float): The 95%
              shot-noise-only fluctuation band for a single bitstring's
              probability at this shot count (see
              `_success_probability_margin_of_error`). A gap between
              `ideal_success_probability` and `noisy_success_probability`
              smaller than this margin is plausibly sampling noise, not
              circuit-noise-induced degradation.
            - "shot_noise_note" (str): Plain-language note stating whether
              the observed success-probability gap exceeds that margin.
            - "estimated_fidelity" (float): Classical (Bhattacharyya-based)
              fidelity between the full ideal and noisy distributions
              (see `_estimate_fidelity`). NOTE: this is a similarity
              measure, not this module's noise-impact metric -- see
              `noise_impact` below for why.
            - "hellinger_distance" (float): Hellinger distance between the
              full distributions, derived from the same Bhattacharyya
              coefficient as `estimated_fidelity` (see
              `_hellinger_distance`). A cross-check on `estimated_fidelity`
              using a roughly-linear (rather than quadratic) scale.
            - "total_variation_distance" (float): Fraction of probability
              mass displaced between the ideal and noisy distributions
              (see `_total_variation_distance`).
            - "noise_impact" (float): This module's primary noise-impact
              metric, defined as `total_variation_distance` rather than
              `1 - estimated_fidelity`. Rationale: because
              `estimated_fidelity` is quadratic in the Bhattacharyya
              coefficient near 1, `1 - estimated_fidelity` compresses
              small-to-moderate distributional differences (e.g. 0.998
              fidelity -> 0.002 "impact", even when the underlying
              perturbation is not negligible). TVD scales roughly
              linearly with the same perturbation and is bounded and
              directly interpretable ("this fraction of measurement
              outcomes was displaced by noise"), making it more
              discriminating in the low-noise regime this project
              targets.
            - "noise_assessment" (str): "LOW" / "MEDIUM" / "HIGH",
              classified from `noise_impact` against
              `NOISE_ASSESSMENT_THRESHOLDS`.
            - "noise_assessment_explanation" (str): Plain-language
              explanation of `noise_assessment`, generated from the
              computed `noise_impact` value.
            - "circuit_reliability" (str): "HIGH" / "MEDIUM" / "LOW",
              classified from `estimated_fidelity` against
              `RELIABILITY_THRESHOLDS`.
            - "estimated_reliability_percent" (float): `estimated_fidelity
              * 100`, for direct display.
            - "reliability_explanation" (str): Plain-language explanation
              of `circuit_reliability`, generated from the computed
              `estimated_fidelity` value.

    Raises:
        TypeError: If `qc` is not a QuantumCircuit instance.
        ValueError: If `shots` is not a positive integer.
        CircuitSimulationError: If either simulation fails (e.g. the
            circuit has no measurement instructions).
    """
    if not isinstance(qc, QuantumCircuit):
        raise TypeError(f"Expected a QuantumCircuit instance, got {type(qc).__name__}")

    if shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots}")

    active_noise_model = noise_model if noise_model is not None else build_noise_model()

    ideal_counts = _run_simulation(qc, shots, noise_model=None)
    noisy_counts = _run_simulation(qc, shots, noise_model=active_noise_model)

    if not ideal_counts:
        raise CircuitSimulationError(
            f"Circuit '{qc.name}' produced no measurement counts. "
            "Does it contain measurement instructions?"
        )

    success_bitstring = max(ideal_counts, key=ideal_counts.get)
    ideal_success_probability = ideal_counts.get(success_bitstring, 0) / shots
    noisy_success_probability = noisy_counts.get(success_bitstring, 0) / shots

    margin_of_error = _success_probability_margin_of_error(shots)
    success_probability_gap = abs(ideal_success_probability - noisy_success_probability)
    shot_noise_note = _build_shot_noise_note(success_probability_gap, margin_of_error)

    bhattacharyya_coefficient = _bhattacharyya_coefficient(ideal_counts, noisy_counts, shots)
    estimated_fidelity = _estimate_fidelity(bhattacharyya_coefficient)
    hellinger_distance = _hellinger_distance(bhattacharyya_coefficient)
    total_variation_distance = _total_variation_distance(ideal_counts, noisy_counts, shots)

    noise_impact = total_variation_distance
    noise_assessment = _classify_noise_assessment(noise_impact)
    noise_assessment_explanation = _build_noise_assessment_explanation(
        noise_impact, noise_assessment
    )

    circuit_reliability = _classify_circuit_reliability(estimated_fidelity)
    reliability_explanation = _build_reliability_explanation(
        estimated_fidelity, circuit_reliability
    )

    return {
        "total_shots": shots,
        "ideal_counts": ideal_counts,
        "noisy_counts": noisy_counts,
        "success_bitstring": success_bitstring,
        "ideal_success_probability": ideal_success_probability,
        "noisy_success_probability": noisy_success_probability,
        "success_probability_margin_of_error": margin_of_error,
        "shot_noise_note": shot_noise_note,
        "estimated_fidelity": estimated_fidelity,
        "hellinger_distance": hellinger_distance,
        "total_variation_distance": total_variation_distance,
        "noise_impact": noise_impact,
        "noise_assessment": noise_assessment,
        "noise_assessment_explanation": noise_assessment_explanation,
        "circuit_reliability": circuit_reliability,
        "estimated_reliability_percent": estimated_fidelity * 100,
        "reliability_explanation": reliability_explanation,
    }
