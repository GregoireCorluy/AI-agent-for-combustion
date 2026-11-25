from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

# This module provides evaluation metrics for the fine-tuning process.

def parse_prediction(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """ Parse a model prediction string into a dictionary of input parameters."""

    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text.split("\n", 1)[-1].rsplit(fence, 1)[0].strip()
    # Thinking is disabled in our rendered prompt, but tolerate a model that
    # still emits an explicit think block before its JSON answer.
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc.msg}"
    if not isinstance(result, dict) or set(result) != {"input_parameters"}:
        return None, "expected_only_input_parameters"
    params = result["input_parameters"]
    if not isinstance(params, dict):
        return None, "input_parameters_must_be_an_object"
    from CombustionAgent.parameters import InputParameters
    expected_fields = set(InputParameters.model_fields)
    if set(params) != expected_fields:
        return params, "schema_invalid: InputParameters field set mismatch"
    try:
        InputParameters.model_validate(params)
    except ValidationError as exc:
        return params, "schema_invalid: " + str(exc)
    return params, None


def _prf(predicted: list[str] | None, expected: list[str] | None) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score for two sets of strings."""

    pred, ref = set(predicted or []), set(expected or [])
    true_positive = len(pred & ref)
    precision = true_positive / len(pred) if pred else float(not ref)
    recall = true_positive / len(ref) if ref else float(not pred)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score(prediction: str, record: dict[str, Any]) -> dict[str, Any]:
    """ Score a single model prediction against the expected input parameters from a record."""
    
    expected_full = json.loads(
        next(m["content"] for m in record["messages"] if m["role"] == "assistant")
    )
    expected = expected_full["input_parameters"]
    params, error = parse_prediction(prediction)
    row: dict[str, Any] = {
        "valid_json": not (error is not None and error.startswith("invalid_json")),
        "schema_valid": params is not None and error is None,
        "parse_error": error,
        "application_family": record.get("metadata", {}).get("application_family", "unknown"),
    }
    if params is None:
        return row
    for name in ("target_species", "retained_species"):
        precision, recall, f1 = _prf(params.get(name), expected.get(name))
        row[f"{name}_precision"] = precision
        row[f"{name}_recall"] = recall
        row[f"{name}_f1"] = f1
    anchors = expected_full.get("explicit_anchors", {})
    checks = []
    for name, value in anchors.items():
        if name.endswith("_unit"):
            checks.append(params.get(name) == value)
        elif name in {"pressure", "temperature", "equivalence_ratio"}:
            start, end = params.get(name + "_start"), params.get(name + "_end")
            checks.append(
                isinstance(start, (int, float))
                and isinstance(end, (int, float))
                and start <= value <= end # Check that explicitly mentioned values are englobed by the range provided by the model. If the model does not provide a range, we consider this check to be failed.
            )
    row["explicit_anchor_preservation"] = (
        sum(checks) / len(checks) if checks else None # If there are no explicit anchors, we don't check this property. Otherwise, we compute the fraction of anchors that are preserved in the model's prediction.
    )
    row["mechanism_null_when_unsupplied"] = (
        params.get("mechanism") is None if expected.get("mechanism") is None else None # If the expected mechanism is None, check that the predicted mechanism is also None. Otherwise, we don't check this property.
    )
    for field in ("pressure", "temperature", "equivalence_ratio"):
        start, end = params.get(field + "_start"), params.get(field + "_end")
        row[field + "_range_sane"] = start is None or end is None or start <= end # Sanity check for the range of pressure, temperature, and equivalence ratio that the model predicts. If the start is greater than the end, it is considered an error.
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate evaluation metrics over a list of scored rows."""

    result: dict[str, Any] = {"examples": len(rows)}
    result["valid_json_rate"] = sum(bool(r["valid_json"]) for r in rows) / max(1, len(rows))
    result["schema_valid_rate"] = sum(bool(r["schema_valid"]) for r in rows) / max(1, len(rows))
    keys = sorted({
        key for row in rows for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"valid_json", "schema_valid"}
    })
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        result[key] = sum(values) / len(values) if values else None
    return result
