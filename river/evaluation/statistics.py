"""Statistical tests for normalized RIVER evaluations."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from statistics import mean, stdev
from typing import Any, Callable

from .metrics import EvaluationError, SCALAR_METRICS, revision_correct, validate_single_comparison_mode


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value))


def _to_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if _is_number(value):
        return float(value)
    return None


def _clean_paired_values(left: list[Any], right: list[Any]) -> tuple[list[float], list[float]]:
    if len(left) != len(right):
        raise EvaluationError("paired values must have the same length")
    left_values: list[float] = []
    right_values: list[float] = []
    for a, b in zip(left, right, strict=True):
        left_number = _to_numeric(a)
        right_number = _to_numeric(b)
        if left_number is not None and right_number is not None:
            left_values.append(left_number)
            right_values.append(right_number)
    return left_values, right_values


def mean_difference(left: list[Any], right: list[Any]) -> float:
    left_values, right_values = _clean_paired_values(left, right)
    if not left_values:
        return math.nan
    return mean(a - b for a, b in zip(left_values, right_values, strict=True))


def paired_bootstrap_ci(
    left: list[Any],
    right: list[Any],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 13,
    statistic: Callable[[list[float], list[float]], float] = mean_difference,
) -> dict[str, float | int]:
    """Bootstrap a paired statistic by resampling aligned units."""

    left_values, right_values = _clean_paired_values(left, right)
    n = len(left_values)
    if n == 0:
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        estimates.append(statistic([left_values[i] for i in indices], [right_values[i] for i in indices]))
    estimates.sort()
    low_index = max(0, min(len(estimates) - 1, int((alpha / 2) * len(estimates))))
    high_index = max(0, min(len(estimates) - 1, int((1 - alpha / 2) * len(estimates)) - 1))
    return {
        "estimate": statistic(left_values, right_values),
        "ci_low": estimates[low_index],
        "ci_high": estimates[high_index],
        "n": n,
    }


def _binomial_two_sided_p(k: int, n: int) -> float:
    if n == 0:
        return math.nan
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_test(left_correct: list[Any], right_correct: list[Any]) -> dict[str, float | int]:
    """Exact McNemar test over paired binary correctness values."""

    left, right = _clean_binary_pairs(left_correct, right_correct)
    b = sum(a and not c for a, c in zip(left, right, strict=True))
    c = sum((not a) and c for a, c in zip(left, right, strict=True))
    statistic = ((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) else 0.0
    return {
        "b_left_only": b,
        "c_right_only": c,
        "n": len(left),
        "statistic": statistic,
        "p_value": _binomial_two_sided_p(min(b, c), b + c),
    }


def _clean_binary_pairs(left: list[Any], right: list[Any]) -> tuple[list[bool], list[bool]]:
    if len(left) != len(right):
        raise EvaluationError("paired binary values must have the same length")
    left_values: list[bool] = []
    right_values: list[bool] = []
    for a, b in zip(left, right, strict=True):
        if a is None or b is None:
            continue
        left_values.append(bool(a))
        right_values.append(bool(b))
    return left_values, right_values


def _permutation_distribution(differences: list[float], *, seed: int, n_permutations: int) -> list[float]:
    n = len(differences)
    if n <= 20:
        values = []
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            values.append(mean(sign * diff for sign, diff in zip(signs, differences, strict=True)))
        return values
    rng = random.Random(seed)
    values = []
    for _ in range(n_permutations):
        values.append(mean((1.0 if rng.random() < 0.5 else -1.0) * diff for diff in differences))
    return values


def paired_permutation_test(
    left: list[Any],
    right: list[Any],
    *,
    alternative: str = "two-sided",
    seed: int = 13,
    n_permutations: int = 10000,
) -> dict[str, float | int | str]:
    """Paired sign-flip permutation test for mean difference."""

    left_values, right_values = _clean_paired_values(left, right)
    differences = [a - b for a, b in zip(left_values, right_values, strict=True)]
    if not differences:
        return {"estimate": math.nan, "p_value": math.nan, "n": 0, "alternative": alternative}
    observed = mean(differences)
    distribution = _permutation_distribution(differences, seed=seed, n_permutations=n_permutations)
    if alternative == "greater":
        p_value = sum(value >= observed for value in distribution) / len(distribution)
    elif alternative == "less":
        p_value = sum(value <= observed for value in distribution) / len(distribution)
    elif alternative == "two-sided":
        p_value = sum(abs(value) >= abs(observed) for value in distribution) / len(distribution)
    else:
        raise ValueError("alternative must be one of: two-sided, greater, less")
    return {"estimate": observed, "p_value": p_value, "n": len(differences), "alternative": alternative}


def effect_sizes(left: list[Any], right: list[Any]) -> dict[str, float | int]:
    left_values, right_values = _clean_paired_values(left, right)
    differences = [a - b for a, b in zip(left_values, right_values, strict=True)]
    if not differences:
        return {"mean_difference": math.nan, "cohen_dz": math.nan, "n": 0}
    diff_mean = mean(differences)
    diff_sd = stdev(differences) if len(differences) > 1 else 0.0
    return {
        "mean_difference": diff_mean,
        "cohen_dz": diff_mean / diff_sd if diff_sd else math.nan,
        "n": len(differences),
    }


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    clean = [(key, value) for key, value in p_values.items() if _is_number(value)]
    m = len(clean)
    adjusted: dict[str, float] = {key: math.nan for key in p_values}
    running_max = 0.0
    for rank, (key, value) in enumerate(sorted(clean, key=lambda item: item[1]), start=1):
        corrected = min(1.0, (m - rank + 1) * value)
        running_max = max(running_max, corrected)
        adjusted[key] = running_max
    return adjusted


def benjamini_hochberg_correction(p_values: dict[str, float]) -> dict[str, float]:
    clean = sorted([(key, value) for key, value in p_values.items() if _is_number(value)], key=lambda item: item[1])
    m = len(clean)
    adjusted: dict[str, float] = {key: math.nan for key in p_values}
    running_min = 1.0
    for rank, (key, value) in reversed(list(enumerate(clean, start=1))):
        corrected = min(running_min, value * m / rank)
        running_min = corrected
        adjusted[key] = min(1.0, corrected)
    return adjusted


def align_by_example(
    records: list[dict[str, Any]],
    left_baseline: str,
    right_baseline: str,
    *,
    intervention: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_single_comparison_mode(records)
    left_rows = [
        row for row in records
        if row.get("baseline") == left_baseline and (intervention is None or row.get("intervention") == intervention)
    ]
    right_rows = [
        row for row in records
        if row.get("baseline") == right_baseline and (intervention is None or row.get("intervention") == intervention)
    ]
    left_map = _unique_example_map(left_rows)
    right_map = _unique_example_map(right_rows)
    shared = sorted(set(left_map) & set(right_map))
    if not shared:
        raise EvaluationError(f"no shared examples for {left_baseline!r} and {right_baseline!r}")
    return [left_map[key] for key in shared], [right_map[key] for key in shared]


def _unique_example_map(records: list[dict[str, Any]]) -> dict[tuple[Any, Any], dict[str, Any]]:
    mapping: dict[tuple[Any, Any], dict[str, Any]] = {}
    duplicates: list[tuple[Any, Any]] = []
    for row in records:
        key = (row.get("example_id"), row.get("seed"))
        if key in mapping:
            duplicates.append(key)
        mapping[key] = row
    if duplicates:
        raise EvaluationError(f"duplicate baseline rows for paired statistic units: {duplicates[:5]}")
    return mapping


def paired_correctness(records: list[dict[str, Any]], left_baseline: str, right_baseline: str) -> tuple[list[bool], list[bool]]:
    left_rows, right_rows = align_by_example(records, left_baseline, right_baseline)
    left_correct = [revision_correct(row) for row in left_rows]
    right_correct = [revision_correct(row) for row in right_rows]
    return _clean_binary_pairs(left_correct, right_correct)


def non_inferiority_test(
    candidate: list[Any],
    reference: list[Any],
    *,
    margin: float,
    seed: int = 13,
    n_permutations: int = 10000,
) -> dict[str, float | int | bool]:
    """One-sided paired non-inferiority test for candidate - reference."""

    candidate_values, reference_values = _clean_paired_values(candidate, reference)
    differences = [a - b for a, b in zip(candidate_values, reference_values, strict=True)]
    if not differences:
        return {
            "observed_difference": math.nan,
            "margin": margin,
            "p_value": math.nan,
            "non_inferior": False,
            "n": 0,
        }
    shifted = [diff + margin for diff in differences]
    observed = mean(shifted)
    distribution = _permutation_distribution(shifted, seed=seed, n_permutations=n_permutations)
    p_value = sum(value >= observed for value in distribution) / len(distribution)
    return {
        "observed_difference": mean(differences),
        "margin": margin,
        "p_value": p_value,
        "non_inferior": observed > 0 and p_value < 0.05,
        "n": len(differences),
    }


def genuine_update_non_inferiority(
    records: list[dict[str, Any]],
    *,
    candidate_baseline: str,
    reference_baseline: str,
    margin: float,
    seed: int = 13,
) -> dict[str, float | int | bool]:
    candidate_rows, reference_rows = align_by_example(
        records,
        candidate_baseline,
        reference_baseline,
        intervention="supported_genuine_update",
    )
    paired_rows = [
        (candidate, reference)
        for candidate, reference in zip(candidate_rows, reference_rows, strict=True)
        if candidate.get("intervention_hypothesis_id") is not None
        and candidate.get("gold_hypothesis_id") == candidate.get("intervention_hypothesis_id")
        and reference.get("intervention_hypothesis_id") is not None
        and reference.get("gold_hypothesis_id") == reference.get("intervention_hypothesis_id")
    ]
    candidate_values = [revision_correct(row) for row, _ in paired_rows]
    reference_values = [revision_correct(row) for _, row in paired_rows]
    return non_inferiority_test(candidate_values, reference_values, margin=margin, seed=seed)


def seed_aggregation(
    records: list[dict[str, Any]],
    metric_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    metric_names = metric_names or ["recency_flip_rate", "genuine_update_accuracy", "unsupported_supersession_rate"]
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row.get("baseline", "unknown"), row.get("comparison_mode", "controlled"), row.get("seed"))].append(row)

    per_seed: dict[tuple[Any, Any], list[dict[str, float]]] = defaultdict(list)
    for (baseline, mode, _seed), items in grouped.items():
        values: dict[str, float] = {}
        for metric in metric_names:
            values[metric] = SCALAR_METRICS[metric](items)
        per_seed[(baseline, mode)].append(values)

    rows: list[dict[str, Any]] = []
    for (baseline, mode), seed_rows in sorted(per_seed.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        row: dict[str, Any] = {"baseline": baseline, "comparison_mode": mode, "n_seeds": len(seed_rows)}
        for metric in metric_names:
            values = [item[metric] for item in seed_rows if _is_number(item[metric])]
            row[f"{metric}_mean"] = mean(values) if values else math.nan
            row[f"{metric}_sd"] = stdev(values) if len(values) > 1 else math.nan
            row[f"{metric}_sem"] = (stdev(values) / math.sqrt(len(values))) if len(values) > 1 else math.nan
        rows.append(row)
    return rows
