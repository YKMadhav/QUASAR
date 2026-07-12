"""
feature_extractor.py
---------------------
Single Responsibility:
    Convert a Qiskit QuantumCircuit into a flat, structured feature
    dictionary intended as input ("X") to a future machine learning
    pipeline.

This module intentionally does NOT:
    - Predict noise, fidelity, or reliability
    - Simulate the circuit or generate labels ("y")
    - Train or run any machine learning model
    - Print anything to the console

Where this fits in the pipeline:
    Parser -> Analyzer -> Feature Extractor -> Noise Simulator (future)
    -> Dataset Generator (future) -> Machine Learning (future)

The dictionary contract here must stay stable and self-contained, since a
future `dataset_generator.py` will pair each feature dict with a simulated
reliability/fidelity label (produced by a future `noise_simulator.py`) to
build ML training rows -- e.g. `{**extract_features(qc), "fidelity": y}`.
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction

# Gate names (as reported by Qiskit's `.name` attribute) treated as
# multi-qubit *entangling* operations. Kept as a module-level, easily
# extensible constant rather than embedding name checks in the extraction
# logic -- adding a new entangling gate later means editing this one set,
# not the function body.
ENTANGLING_GATE_NAMES: frozenset[str] = frozenset({"cx", "cz", "cp", "swap", "ccx"})

# Structural instructions that carry no gate-arity or noise-relevant
# meaning and are excluded from arity/entangling/parameter counting.
_NON_GATE_INSTRUCTIONS: frozenset[str] = frozenset({"barrier"})


def extract_features(
    qc: QuantumCircuit,
    entangling_gate_names: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Extract a structured, ML-ready feature dictionary from a circuit.

    Args:
        qc: A parsed Qiskit QuantumCircuit instance.
        entangling_gate_names: Optional override for which gate names are
            considered "entangling". Defaults to `ENTANGLING_GATE_NAMES`.
            Exposed as a parameter -- rather than hardcoded -- so future
            modules can extend or customize the entangling-gate
            definition without modifying this module's logic.

    Returns:
        A flat dictionary containing:
            - "num_qubits", "num_clbits", "depth", "width",
              "total_operations": general circuit-scale features (int).
            - "single_qubit_gates", "two_qubit_gates",
              "three_qubit_gates": counts derived generically from each
              instruction's operand (qubit) count, not gate-name lookups.
            - "measurement_gates": count of "measure" instructions.
            - "parameterized_gates": count of instructions whose operation
              carries one or more parameters, detected generically via
              `instruction.operation.params` (works for any bound or
              unbound parameterized gate, not just RX/RY/RZ/CP).
            - "entangling_gates": count of instructions whose gate name is
              in `entangling_gate_names`.
            - "gate_distribution" (dict[str, int]): count of every
              distinct operation name present in the circuit, gate names
              are never hardcoded or filtered.

    Raises:
        TypeError: If `qc` is not a QuantumCircuit instance.
    """
    if not isinstance(qc, QuantumCircuit):
        raise TypeError(
            f"Expected a QuantumCircuit instance, got {type(qc).__name__}"
        )

    entangling_names = entangling_gate_names or ENTANGLING_GATE_NAMES
    gate_distribution: dict[str, int] = dict(qc.count_ops())

    single_qubit_gates = 0
    two_qubit_gates = 0
    three_qubit_gates = 0
    parameterized_gates = 0
    entangling_gates = 0

    for instruction in qc.data:
        name = instruction.operation.name

        if name == "measure" or name in _NON_GATE_INSTRUCTIONS:
            continue

        arity = _operand_count(instruction)
        if arity == 1:
            single_qubit_gates += 1
        elif arity == 2:
            two_qubit_gates += 1
        elif arity == 3:
            three_qubit_gates += 1

        if _is_parameterized(instruction):
            parameterized_gates += 1

        if name in entangling_names:
            entangling_gates += 1

    return {
        "num_qubits": qc.num_qubits,
        "num_clbits": qc.num_clbits,
        "depth": qc.depth(),
        "width": qc.width(),
        "total_operations": qc.size(),
        "single_qubit_gates": single_qubit_gates,
        "two_qubit_gates": two_qubit_gates,
        "three_qubit_gates": three_qubit_gates,
        "measurement_gates": gate_distribution.get("measure", 0),
        "parameterized_gates": parameterized_gates,
        "entangling_gates": entangling_gates,
        "gate_distribution": gate_distribution,
    }


def _operand_count(instruction: CircuitInstruction) -> int:
    """Return the number of qubits a single instruction acts on."""
    return len(instruction.qubits)


def _is_parameterized(instruction: CircuitInstruction) -> bool:
    """Return True if the instruction's operation carries any parameters.

    Checks `operation.params` generically so any gate with bound
    (numeric) or unbound (symbolic `Parameter`) arguments is detected --
    RX/RY/RZ/CP today, and any future parameterized gate -- without
    maintaining a name-based lookup list.
    """
    return len(instruction.operation.params) > 0
