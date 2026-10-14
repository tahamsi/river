"""Validation-only calibration utilities for verification thresholds."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


VALIDATION_SPLITS = {"validation", "valid", "dev"}


def calibrate_support_threshold(
    rows: Iterable[dict[str, Any]],
    *,
    split_field: str = "split",
    score_field: str = "supported_score",
    label_field: str = "label",
) -> float:
    """Choose a support threshold from validation/dev rows only."""

    validation_rows = [row for row in rows if row.get(split_field) in VALIDATION_SPLITS]
    if not validation_rows:
        raise ValueError("calibration requires validation/dev rows")
    disallowed = [row.get(split_field) for row in rows if row.get(split_field) not in VALIDATION_SPLITS]
    if disallowed:
        raise ValueError("calibration input must not include train/test rows")

    candidates = sorted({float(row[score_field]) for row in validation_rows})
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in candidates:
        correct = 0
        for row in validation_rows:
            predicted = float(row[score_field]) >= threshold
            gold = str(row[label_field]).lower() in {"supported", "entailed", "entailment", "1", "true"}
            correct += predicted == gold
        accuracy = correct / len(validation_rows)
        if accuracy > best_accuracy or (accuracy == best_accuracy and threshold > best_threshold):
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold
