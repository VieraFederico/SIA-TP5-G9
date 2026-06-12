# utils/csv_utils.py
import csv
import json
from pathlib import Path
from typing import Iterable, Any


def _as_json_str(value: Any) -> str:
    """Serialize lists/arrays/scalars to a JSON string for CSV storage."""
    try:
        return json.dumps(value)
    except TypeError:
        # Fallback for numpy arrays or other iterables
        if hasattr(value, "tolist"):
            return json.dumps(value.tolist())
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value))
        return json.dumps(str(value))


def append_perceptron_result(
    output_path: str | Path,
    perceptron_type: str,
    architecture: list[int] | None,
    learning_rate: float,
    beta_value: float | None,
    weights,
    bias,
    best_train_error: float | None,
    best_eval_error: float | None,
) -> None:
    """
    Append a result row to CSV. Creates the file with header if it doesn't exist.

    Args:
        beta_value:
        learning_rate:
        output_path: destination CSV path
        perceptron_type: "linear" | "non_linear" | "multilayer"
        architecture: e.g. [7, 4, 1] or None for single-layer
        weights: weights array/list
        bias: bias scalar or array
        best_train_error: last/lowest training error
        best_eval_error: last/lowest eval error (if available)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "perceptron_type",
        "architecture",
        "learning_rate",
        "beta_value",
        "weights",
        "bias",
        "best_train_error",
        "best_eval_error",
    ]

    row = {
        "perceptron_type": perceptron_type,
        "architecture": _as_json_str(architecture),
        "learning_rate": learning_rate,
        "beta_value": beta_value if beta_value is not None else "",
        "weights": _as_json_str(weights),
        "bias": _as_json_str(bias),
        "best_train_error": "" if best_train_error is None else f"{best_train_error:.3f}",
        "best_eval_error": "" if best_eval_error is None else f"{best_eval_error:.3f}",
    }

    write_header = not output_path.exists()
    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)