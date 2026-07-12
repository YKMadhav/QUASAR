"""
parser.py
---------
Single Responsibility:
    Convert an OpenQASM 3 (.qasm) file on disk into a Qiskit `QuantumCircuit`
    object.

This module intentionally does NOT:
    - Analyze circuit structure
    - Print anything to the console
    - Perform simulation, optimization, or transpilation

Any future change to *how* circuits are loaded (different QASM version,
remote circuit source, different SDK) should be isolated to this module.
"""

from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.qasm3 import load


class QasmParsingError(Exception):
    """Raised when a .qasm file exists but cannot be parsed into a valid
    QuantumCircuit (e.g. malformed OpenQASM 3 syntax).

    Kept as a project-specific exception (rather than surfacing the raw
    Qiskit exception) so that callers -- CLI, future API, future batch
    runner -- can catch a single, stable error type regardless of which
    underlying parsing library or Qiskit version is in use.
    """


def load_qasm_file(file_path: str | Path) -> QuantumCircuit:
    """Load an OpenQASM 3 file from disk and return it as a QuantumCircuit.

    Args:
        file_path: Path to a .qasm file containing OpenQASM 3 source code.

    Returns:
        QuantumCircuit: The parsed quantum circuit.

    Raises:
        FileNotFoundError: If no file exists at `file_path`.
        ValueError: If `file_path` does not have a `.qasm` extension.
        QasmParsingError: If the file exists but is not valid OpenQASM 3.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"QASM file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, but got a directory: {path}")

    if path.suffix.lower() != ".qasm":
        raise ValueError(
            f"Expected a file with a '.qasm' extension, got: {path.suffix}"
        )

    try:
        circuit = load(str(path))
    except OSError as exc:
        raise QasmParsingError(
            f"Could not read file '{path}': {exc}"
        ) from exc
    except Exception as exc:
        # The OpenQASM 3 parsing toolchain (qiskit.qasm3 -> qiskit_qasm3_import
        # -> openqasm3) can raise several distinct exception types depending
        # on where a malformed program fails (lexing, parsing, semantic
        # conversion to a QuantumCircuit). Rather than coupling this module
        # to every possible upstream exception class -- which would break
        # silently on a dependency upgrade -- we treat any failure at this
        # boundary as "invalid QASM content" and normalize it to a single,
        # stable, project-level exception type.
        raise QasmParsingError(
            f"Failed to parse OpenQASM 3 content in '{path}': {exc}"
        ) from exc

    return circuit
