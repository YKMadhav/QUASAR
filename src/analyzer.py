"""
analyzer.py
-----------
Single Responsibility:
    Inspect a Qiskit `QuantumCircuit` object and extract purely structural,
    factual information about it (no interpretation, no scoring, no
    prediction).

This module intentionally does NOT:
    - Simulate the circuit
    - Estimate fidelity or noise
    - Use machine learning
    - Print anything to the console

The dictionary returned here is the "raw facts" contract that future
modules (reliability scoring, explainability, ML feature extraction,
dashboards) will consume. Keeping it a plain, flat dict keeps it trivially
serializable to JSON for a future API layer.
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

# Name Qiskit gives circuits that were never explicitly named.
_UNNAMED_CIRCUIT_LABEL = "circuit-"


def analyze_circuit(circuit: QuantumCircuit) -> dict[str, Any]:
    """Extract structural information from a QuantumCircuit.

    Args:
        circuit: A parsed Qiskit QuantumCircuit instance.

    Returns:
        A dictionary with the following keys:
            - "name" (str | None): Circuit name, or None if auto-generated.
            - "num_qubits" (int): Number of qubits.
            - "num_clbits" (int): Number of classical bits.
            - "depth" (int): Circuit depth.
            - "width" (int): Circuit width (qubits + clbits).
            - "total_operations" (int): Total instruction count.
            - "gate_counts" (dict[str, int]): Count of each operation type,
              keyed by lowercase gate name (e.g. "h", "cx", "measure").
            - "gates_used" (list[str]): Sorted list of distinct gate names
              used in the circuit (excludes "measure").
            - "num_measurements" (int): Number of measurement operations.

    Raises:
        TypeError: If `circuit` is not a QuantumCircuit instance.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            f"Expected a QuantumCircuit instance, got {type(circuit).__name__}"
        )

    gate_counts: dict[str, int] = dict(circuit.count_ops())
    num_measurements = gate_counts.get("measure", 0)
    gates_used = sorted(name for name in gate_counts if name != "measure")

    return {
        "name": _resolve_circuit_name(circuit),
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "depth": circuit.depth(),
        "width": circuit.width(),
        "total_operations": circuit.size(),
        "gate_counts": gate_counts,
        "gates_used": gates_used,
        "num_measurements": num_measurements,
    }


def _resolve_circuit_name(circuit: QuantumCircuit) -> str | None:
    """Return the circuit's explicit name, or None if it was auto-generated.

    Qiskit assigns an auto-generated name like "circuit-123" to any
    QuantumCircuit that wasn't given an explicit `name=` at construction
    time. We surface `None` in that case so downstream consumers (e.g. the
    CLI report) can decide how to display "no name" rather than showing a
    meaningless auto-generated id.
    """
    if circuit.name and not circuit.name.startswith(_UNNAMED_CIRCUIT_LABEL):
        return circuit.name
    return None
