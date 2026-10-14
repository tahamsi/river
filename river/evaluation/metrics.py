"""Normalized RIVER evaluation metrics.

The functions in this module operate on serialized ``PredictionRecord`` rows.
They deliberately skip rows with missing labels instead of treating missing
values as negative examples or zero-valued scores.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Callable, Iterable


class EvaluationError(ValueError):
    """Raised when an analysis would combine incompatible or invalid records."""


ABSTAIN = "abstain"
CONFLICT = "disputed"
EVOLUTION = "evolution"
CALIBRATION_SPLITS = {"calibration", "dev", "validation", "val"}
TEST_SPLITS = {"test"}
# retroactive_correction contains evidence text that the source audit could not
# anchor in the cited record, and has too few pairs to support an interval. It
# is reported as a case study with counts and is excluded from every aggregate.
# supported_genuine_update and temporal_evolution were reviewed row by row and
# approved on 2026-09-18, and are included. See docs/integrity.md.
QUALITATIVE_ONLY_INTERVENTIONS = {
    "retroactive_correction",
}


PAIRED_INTERVENTIONS = {
    "order_reversal",
    "expired_evidence",
    "unsupported_late_assertion",
    "repeated_weak_assertion",
    "supported_genuine_update",
    "retroactive_correction",
    "temporal_evolution",
    "concurrent_disagreement",
    "evidence_removal",
    "insufficient_evidence_nei",
}


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_float(value: Any) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _mean_or_nan(values: Iterable[float | bool | None]) -> float:
    clean = [float(value) for value in values if value is not None and not _is_missing(value)]
    return mean(clean) if clean else math.nan


def _safe_div(numerator: float, denominator: float, *, undefined: float = math.nan) -> float:
    return numerator / denominator if denominator else undefined


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _transformation_metadata(row: dict[str, Any]) -> dict[str, Any]:
    direct = row.get("transformation_metadata")
    if isinstance(direct, dict):
        return direct
    metadata = _metadata(row)
    nested = metadata.get("transformation_metadata")
    return nested if isinstance(nested, dict) else {}


def _variant_role(row: dict[str, Any]) -> str:
    return str(_transformation_metadata(row).get("variant_role", "")).lower()


def _state(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    return str(value).lower() if value is not None else None


def revision_correct(row: dict[str, Any]) -> bool | None:
    """Return state/hypothesis correctness when the required labels are present."""

    gold_state = _state(row, "gold_state")
    predicted_state = _state(row, "predicted_state")
    if gold_state is None or predicted_state is None:
        return None
    if predicted_state != gold_state:
        return False
    gold_hypothesis = row.get("gold_hypothesis_id")
    if gold_hypothesis is not None:
        return row.get("predicted_hypothesis_id") == gold_hypothesis
    return True


def answer_or_revision_correct(row: dict[str, Any]) -> bool | None:
    if row.get("answer_correct") is not None:
        return bool(row["answer_correct"])
    return revision_correct(row)


def validate_required_pair_ids(records: list[dict[str, Any]], interventions: set[str] | None = None) -> None:
    interventions = interventions or PAIRED_INTERVENTIONS
    missing = [
        row.get("example_id", "<unknown>")
        for row in records
        if row.get("intervention") in interventions and not row.get("pair_id")
    ]
    if missing:
        raise EvaluationError(f"paired analysis requires pair_id for examples: {missing}")


def validate_single_comparison_mode(records: list[dict[str, Any]]) -> str | None:
    modes = {row.get("comparison_mode", "controlled") for row in records if row.get("comparison_mode") is not None}
    if len(modes) > 1:
        raise EvaluationError(
            "refusing to combine incompatible controlled/native comparisons; "
            f"filter or group by comparison_mode first: {sorted(modes)}"
        )
    return next(iter(modes)) if modes else None


def validate_no_calibration_test_mix(records: list[dict[str, Any]]) -> None:
    splits = {str(row.get("split", "")).lower() for row in records if row.get("split") is not None}
    has_test = bool(splits & TEST_SPLITS)
    has_calibration = bool(splits & CALIBRATION_SPLITS)
    if has_test and has_calibration:
        raise EvaluationError(
            "refusing final test statistics with mixed calibration/dev/validation and test splits; "
            "filter to test split or run calibration analysis separately"
        )


def _pair_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("pair_id"),
        row.get("baseline"),
        row.get("comparison_mode", "controlled"),
        row.get("seed"),
        row.get("run_id"),
    )


def paired_records(records: list[dict[str, Any]], intervention: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    relevant = [row for row in records if row.get("intervention") == intervention]
    if not relevant:
        return []
    validate_required_pair_ids(relevant, {intervention})
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in relevant:
        grouped[_pair_group_key(row)].append(row)

    failures: list[str] = []
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(items) != 2:
            failures.append(f"{key[0]} seed={key[3]} run={key[4]} expected 2 variants, found {len(items)}")
            continue
        left, right = sorted(items, key=lambda row: str(row.get("example_id", "")))
        pairs.append((left, right))
    if failures:
        raise EvaluationError("\n".join(failures))
    return pairs


def _prediction_signature(row: dict[str, Any]) -> tuple[Any, Any]:
    """What this system answered, in a form comparable across a pair.

    A system that selects a hypothesis reports it. One that only writes prose
    has its answer read back out by the scoring pass, so its predictions can be
    compared across arrival orders instead of being uniformly empty.
    """

    return (_state(row, "predicted_state"), _selected_hypothesis(row))


def _selected_hypothesis(row: dict[str, Any]) -> Any:
    """The hypothesis this system settled on, however it expressed it."""

    selected = row.get("predicted_hypothesis_id")
    return row.get("answer_named_hypothesis_id") if selected is None else selected


def _expressed_an_answer(row: dict[str, Any]) -> bool:
    return _prediction_signature(row) != (None, None)


def _gold_signature(row: dict[str, Any]) -> tuple[Any, Any]:
    return (_state(row, "gold_state"), row.get("gold_hypothesis_id"))


def _pair_correctness(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[bool | None, bool | None]:
    return revision_correct(pair[0]), revision_correct(pair[1])


def _rows_for_intervention(records: list[dict[str, Any]], intervention: str) -> list[dict[str, Any]]:
    return [row for row in records if row.get("intervention") == intervention]


def _variant_rows(records: list[dict[str, Any]], intervention: str, role_tokens: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = _rows_for_intervention(records, intervention)
    role_rows = [row for row in rows if any(token in _variant_role(row) for token in role_tokens)]
    return role_rows or rows


def recency_flip_rate(records: list[dict[str, Any]]) -> float:
    # expired_evidence reverses arrival order too; its distractor has merely
    # expired as well, so a flip there is the same failure being measured.
    pairs = paired_records(records, "order_reversal") + paired_records(records, "expired_evidence")
    # A pair where neither variant expressed an answer held nothing stable, so
    # counting it as "did not flip" would reward silence with the best score.
    countable = [
        pair for pair in pairs
        if _expressed_an_answer(pair[0]) or _expressed_an_answer(pair[1])
    ]
    return _mean_or_nan(_prediction_signature(a) != _prediction_signature(b) for a, b in countable)


def order_reversal_consistency(records: list[dict[str, Any]]) -> float:
    value = recency_flip_rate(records)
    return math.nan if math.isnan(value) else 1.0 - value


def _normalize_distribution(distribution: dict[str, Any]) -> dict[str, float] | None:
    values = {str(key): _as_float(value) for key, value in distribution.items()}
    if not values or any(value is None for value in values.values()):
        return None
    total = sum(float(value) for value in values.values())
    if total <= 0:
        return None
    return {key: float(value) / total for key, value in values.items() if value is not None}


def _js_divergence(left: dict[str, float], right: dict[str, float]) -> float | None:
    if set(left) != set(right):
        return None
    keys = sorted(left)
    midpoint = {key: 0.5 * (left[key] + right[key]) for key in keys}

    def kl(p: dict[str, float], q: dict[str, float]) -> float:
        total = 0.0
        for key in keys:
            if p[key] > 0:
                total += p[key] * math.log2(p[key] / q[key])
        return total

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def probability_recency_sensitivity(records: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for left, right in paired_records(records, "order_reversal"):
        left_probs = left.get("state_probabilities")
        right_probs = right.get("state_probabilities")
        if not isinstance(left_probs, dict) or not isinstance(right_probs, dict):
            continue
        left_distribution = _normalize_distribution(left_probs)
        right_distribution = _normalize_distribution(right_probs)
        if left_distribution is None or right_distribution is None:
            continue
        divergence = _js_divergence(left_distribution, right_distribution)
        if divergence is not None:
            values.append(divergence)
    return _mean_or_nan(values)


def unsupported_supersession_rate(records: list[dict[str, Any]]) -> float:
    rows = _variant_rows(records, "unsupported_late_assertion", ("late_unsupported", "unsupported_assertion"))
    return _mean_or_nan(
        _selected_hypothesis(row) == row.get("intervention_hypothesis_id")
        for row in rows
        if _selected_hypothesis(row) is not None and row.get("intervention_hypothesis_id") is not None
    )


def repetition_resistance(records: list[dict[str, Any]]) -> float:
    rows = _variant_rows(records, "repeated_weak_assertion", ("repeated", "weak"))
    superseded = [
        _selected_hypothesis(row) == row.get("intervention_hypothesis_id")
        for row in rows
        if _selected_hypothesis(row) is not None and row.get("intervention_hypothesis_id") is not None
    ]
    value = _mean_or_nan(superseded)
    return math.nan if math.isnan(value) else 1.0 - value


def _order_pair_categories(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for pair in paired_records(records, "order_reversal"):
        left, right = pair
        left_correct, right_correct = _pair_correctness(pair)
        if left_correct is None or right_correct is None:
            continue
        stable = _prediction_signature(left) == _prediction_signature(right)
        if stable and left_correct and right_correct:
            counts["stable_and_correct"] += 1
        elif stable and not left_correct and not right_correct:
            counts["stable_but_wrong"] += 1
        elif not stable and not left_correct and right_correct:
            counts["flipped_to_correct"] += 1
        elif not stable and left_correct and not right_correct:
            counts["flipped_to_wrong"] += 1
        else:
            counts["other"] += 1
    return counts


def _order_category_rate(records: list[dict[str, Any]], category: str) -> float:
    counts = _order_pair_categories(records)
    total = sum(counts.values())
    return counts[category] / total if total else math.nan


def stable_and_correct(records: list[dict[str, Any]]) -> float:
    return _order_category_rate(records, "stable_and_correct")


def stable_but_wrong(records: list[dict[str, Any]]) -> float:
    return _order_category_rate(records, "stable_but_wrong")


def flipped_to_correct(records: list[dict[str, Any]]) -> float:
    return _order_category_rate(records, "flipped_to_correct")


def flipped_to_wrong(records: list[dict[str, Any]]) -> float:
    return _order_category_rate(records, "flipped_to_wrong")


def _accuracy_for_rows(rows: list[dict[str, Any]]) -> float:
    return _mean_or_nan(revision_correct(row) for row in rows)


def genuine_update_accuracy(records: list[dict[str, Any]]) -> float:
    rows = _rows_for_intervention(records, "supported_genuine_update")
    update_rows = [
        row for row in rows
        if row.get("intervention_hypothesis_id") is not None
        and row.get("gold_hypothesis_id") == row.get("intervention_hypothesis_id")
    ]
    return _accuracy_for_rows(update_rows or rows)


def retroactive_correction_accuracy(records: list[dict[str, Any]]) -> float:
    rows = _rows_for_intervention(records, "retroactive_correction")
    correction_rows = [
        row for row in rows
        if row.get("intervention_hypothesis_id") is not None
        and row.get("gold_hypothesis_id") == row.get("intervention_hypothesis_id")
    ]
    return _accuracy_for_rows(correction_rows or rows)


def _temporal_slice(row: dict[str, Any]) -> str | None:
    metadata = _transformation_metadata(row)
    for key in ("temporal_slice", "target_state", "state_time", "time_slice"):
        value = metadata.get(key)
        if value is not None:
            return str(value).lower()
    role = _variant_role(row)
    if "historical" in role or "past" in role:
        return "historical"
    if "current" in role or "present" in role:
        return "current"
    return None


def historical_state_accuracy(records: list[dict[str, Any]]) -> float:
    return _accuracy_for_rows([row for row in records if _temporal_slice(row) == "historical"])


def current_state_accuracy(records: list[dict[str, Any]]) -> float:
    return _accuracy_for_rows([row for row in records if _temporal_slice(row) == "current"])


def stale_belief_rate(records: list[dict[str, Any]]) -> float:
    values: list[bool] = []
    for row in records:
        stale_id = _transformation_metadata(row).get("stale_hypothesis_id")
        if stale_id is None:
            stale_id = _metadata(row).get("stale_hypothesis_id")
        if stale_id is None or row.get("predicted_hypothesis_id") is None:
            continue
        values.append(row.get("predicted_hypothesis_id") == stale_id)
    return _mean_or_nan(values)


def _binary_counts(
    records: list[dict[str, Any]],
    positive_label: str,
    *,
    state_key: str = "predicted_state",
    gold_key: str = "gold_state",
) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0}
    for row in records:
        gold = _state(row, gold_key)
        predicted = _state(row, state_key)
        if gold is None or predicted is None:
            continue
        counts["n"] += 1
        gold_positive = gold == positive_label
        predicted_positive = predicted == positive_label
        if gold_positive and predicted_positive:
            counts["tp"] += 1
        elif not gold_positive and predicted_positive:
            counts["fp"] += 1
        elif gold_positive and not predicted_positive:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def _precision_recall_f1_from_counts(counts: dict[str, int]) -> dict[str, float]:
    if counts.get("n") == 0:
        return {"precision": math.nan, "recall": math.nan, "f1": math.nan}
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = _safe_div(tp, tp + fp, undefined=0.0)
    recall = _safe_div(tp, tp + fn, undefined=math.nan)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and not math.isnan(recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def conflict_precision(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_binary_counts(records, CONFLICT))["precision"]


def conflict_recall(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_binary_counts(records, CONFLICT))["recall"]


def conflict_f1(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_binary_counts(records, CONFLICT))["f1"]


def conflict_specificity(records: list[dict[str, Any]]) -> float:
    counts = _binary_counts(records, CONFLICT)
    return _safe_div(counts["tn"], counts["tn"] + counts["fp"])


def false_conflict_rate(records: list[dict[str, Any]]) -> float:
    specificity = conflict_specificity(records)
    return math.nan if math.isnan(specificity) else 1.0 - specificity


def conflict_macro_f1(records: list[dict[str, Any]]) -> float:
    labels = [CONFLICT, f"not_{CONFLICT}"]
    f1_values: list[float] = []
    for label in labels:
        if label == CONFLICT:
            counts = _binary_counts(records, CONFLICT)
        else:
            base = _binary_counts(records, CONFLICT)
            counts = {"tp": base["tn"], "fp": base["fn"], "tn": base["tp"], "fn": base["fp"], "n": base["n"]}
        if counts["tp"] + counts["fn"] == 0:
            continue
        f1_values.append(_precision_recall_f1_from_counts(counts)["f1"])
    return _mean_or_nan(f1_values)


def conflict_evolution_confusion_matrix(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    labels = [CONFLICT, EVOLUTION, "other"]
    matrix = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    for row in records:
        gold = _state(row, "gold_state")
        predicted = _state(row, "predicted_state")
        if gold is None or predicted is None:
            continue
        gold_label = gold if gold in {CONFLICT, EVOLUTION} else "other"
        predicted_label = predicted if predicted in {CONFLICT, EVOLUTION} else "other"
        matrix[gold_label][predicted_label] += 1
    return matrix


def abstention_precision(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_binary_counts(records, ABSTAIN))["precision"]


def abstention_recall(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_binary_counts(records, ABSTAIN))["recall"]


def abstention_f1(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_binary_counts(records, ABSTAIN))["f1"]


def coverage(records: list[dict[str, Any]]) -> float:
    predictions = [_state(row, "predicted_state") for row in records if _state(row, "predicted_state") is not None]
    if not predictions:
        return math.nan
    return mean(predicted != ABSTAIN for predicted in predictions)


def selective_risk(records: list[dict[str, Any]]) -> float:
    covered = [row for row in records if _state(row, "predicted_state") not in {None, ABSTAIN}]
    correctness = [answer_or_revision_correct(row) for row in covered]
    accuracy = _mean_or_nan(correctness)
    return math.nan if math.isnan(accuracy) else 1.0 - accuracy


def _confidence_correctness(records: list[dict[str, Any]]) -> list[tuple[float, bool]]:
    pairs: list[tuple[float, bool]] = []
    for row in records:
        confidence = _as_float(row.get("confidence"))
        correct = answer_or_revision_correct(row)
        if confidence is None or correct is None:
            continue
        pairs.append((max(0.0, min(1.0, confidence)), bool(correct)))
    return pairs


def risk_coverage_curve(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    pairs = sorted(_confidence_correctness(records), key=lambda item: item[0], reverse=True)
    n = len(pairs)
    if not n:
        return []
    correct = 0
    points: list[dict[str, float]] = []
    for index, (confidence, is_correct) in enumerate(pairs, start=1):
        correct += int(is_correct)
        coverage_value = index / n
        accuracy = correct / index
        points.append({
            "coverage": coverage_value,
            "risk": 1.0 - accuracy,
            "accuracy": accuracy,
            "threshold": confidence,
            "n": index,
        })
    return points


def area_under_risk_coverage_curve(records: list[dict[str, Any]]) -> float:
    points = risk_coverage_curve(records)
    if not points:
        return math.nan
    area = 0.0
    previous_coverage = 0.0
    previous_risk = points[0]["risk"]
    for point in points:
        width = point["coverage"] - previous_coverage
        area += width * (previous_risk + point["risk"]) / 2.0
        previous_coverage = point["coverage"]
        previous_risk = point["risk"]
    return area


def accuracy_at_fixed_coverage(records: list[dict[str, Any]], target_coverage: float = 0.8) -> float:
    pairs = sorted(_confidence_correctness(records), key=lambda item: item[0], reverse=True)
    if not pairs:
        return math.nan
    k = max(1, min(len(pairs), math.ceil(len(pairs) * target_coverage)))
    return mean(correct for _, correct in pairs[:k])


def answer_accuracy(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(bool(row["answer_correct"]) for row in records if row.get("answer_correct") is not None)


def answer_text_accuracy(records: list[dict[str, Any]]) -> float:
    """Accuracy of the explicit direct answer under the decision-render generation contract.

    Defined for every system, unlike answer_accuracy, which needs a selected
    hypothesis and so is undefined for baselines that only generate prose. Rows
    where no single answer is correct carry None and are excluded rather than
    counted as failures.
    """

    return _mean_or_nan(
        bool(row["answer_text_correct"]) for row in records if row.get("answer_text_correct") is not None
    )


def end_to_end_answer_accuracy(records: list[dict[str, Any]]) -> float:
    """Accuracy requiring the structured decision and rendered answer to agree."""

    return _mean_or_nan(
        bool(row["end_to_end_answer_correct"])
        for row in records
        if row.get("end_to_end_answer_correct") is not None
    )


def citation_support_rate(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        bool(row["citation_supported"]) for row in records if row.get("citation_supported") is not None
    )


def _claim_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    results = row.get("claim_verification_results")
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def citation_compliance_rate(records: list[dict[str, Any]]) -> float:
    values: list[bool] = []
    for row in records:
        claims = _claim_results(row)
        if not claims:
            continue
        values.append(all(bool(claim.get("citations")) for claim in claims))
    return _mean_or_nan(values)


def _claim_gold_label(row: dict[str, Any], claim: dict[str, Any]) -> str | None:
    claim_id = claim.get("claim_id")
    direct = claim.get("gold_label")
    if direct is not None:
        return str(direct).lower()
    metadata = claim.get("metadata")
    if isinstance(metadata, dict) and metadata.get("gold_label") is not None:
        return str(metadata["gold_label"]).lower()
    row_metadata = _metadata(row)
    gold_labels = row_metadata.get("claim_gold_labels") or row_metadata.get("citation_gold_labels")
    if isinstance(gold_labels, dict) and claim_id in gold_labels:
        return str(gold_labels[claim_id]).lower()
    transformation = _transformation_metadata(row)
    gold_labels = transformation.get("claim_gold_labels") or transformation.get("citation_gold_labels")
    if isinstance(gold_labels, dict) and claim_id in gold_labels:
        return str(gold_labels[claim_id]).lower()
    claim_text = " ".join(str(claim.get("claim_text") or claim.get("text") or "").split()).casefold()
    if claim_text:
        gold_claims = row_metadata.get("gold_claims") or transformation.get("gold_claims") or []
        for gold_claim in gold_claims:
            if not isinstance(gold_claim, dict):
                continue
            gold_text = " ".join(str(gold_claim.get("text") or "").split()).casefold()
            if gold_text == claim_text and gold_claim.get("gold_label") is not None:
                return str(gold_claim["gold_label"]).lower()
    return None


def _claim_label_pairs(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for row in records:
        for claim in _claim_results(row):
            predicted = claim.get("label")
            gold = _claim_gold_label(row, claim)
            if predicted is None or gold is None:
                continue
            pairs.append((str(predicted).lower(), gold))
    return pairs


def _label_counts(records: list[dict[str, Any]], positive_label: str) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0}
    for predicted, gold in _claim_label_pairs(records):
        counts["n"] += 1
        predicted_positive = predicted == positive_label
        gold_positive = gold == positive_label
        if predicted_positive and gold_positive:
            counts["tp"] += 1
        elif predicted_positive and not gold_positive:
            counts["fp"] += 1
        elif not predicted_positive and gold_positive:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def citation_support_precision(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_label_counts(records, "supported"))["precision"]


def citation_support_recall(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_label_counts(records, "supported"))["recall"]


def citation_support_f1(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_label_counts(records, "supported"))["f1"]


def contradiction_f1(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_label_counts(records, "contradicted"))["f1"]


def neutral_insufficient_f1(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_label_counts(records, "neutral"))["f1"]


def temporal_mismatch_f1(records: list[dict[str, Any]]) -> float:
    return _precision_recall_f1_from_counts(_label_counts(records, "temporal_mismatch"))["f1"]


def supported_and_correct(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        bool(row["citation_supported"]) and bool(row["answer_correct"])
        for row in records
        if row.get("citation_supported") is not None and row.get("answer_correct") is not None
    )


def compliant_but_unsupported(records: list[dict[str, Any]]) -> float:
    values: list[bool] = []
    for row in records:
        if row.get("citation_supported") is None:
            continue
        claims = _claim_results(row)
        if not claims:
            continue
        compliant = all(bool(claim.get("citations")) for claim in claims)
        values.append(compliant and not bool(row["citation_supported"]))
    return _mean_or_nan(values)


def supported_but_wrong(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        bool(row["citation_supported"]) and not bool(row["answer_correct"])
        for row in records
        if row.get("citation_supported") is not None and row.get("answer_correct") is not None
    )


def supported_and_text_correct(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        bool(row["citation_supported"]) and bool(row["answer_text_correct"])
        for row in records
        if row.get("citation_supported") is not None and row.get("answer_text_correct") is not None
    )


def supported_but_text_wrong(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        bool(row["citation_supported"]) and not bool(row["answer_text_correct"])
        for row in records
        if row.get("citation_supported") is not None and row.get("answer_text_correct") is not None
    )


def brier_score(records: list[dict[str, Any]]) -> float:
    pairs = _confidence_correctness(records)
    return _mean_or_nan((confidence - float(correct)) ** 2 for confidence, correct in pairs)


def reliability_bins(records: list[dict[str, Any]], n_bins: int = 10) -> list[dict[str, float | int]]:
    pairs = _confidence_correctness(records)
    if not pairs:
        return []
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        if index == n_bins - 1:
            members = [(conf, correct) for conf, correct in pairs if low <= conf <= high]
        else:
            members = [(conf, correct) for conf, correct in pairs if low <= conf < high]
        if not members:
            bins.append({
                "bin": index,
                "low": low,
                "high": high,
                "n": 0,
                "confidence": math.nan,
                "accuracy": math.nan,
                "gap": math.nan,
            })
            continue
        avg_confidence = mean(conf for conf, _ in members)
        accuracy = mean(correct for _, correct in members)
        bins.append({
            "bin": index,
            "low": low,
            "high": high,
            "n": len(members),
            "confidence": avg_confidence,
            "accuracy": accuracy,
            "gap": abs(avg_confidence - accuracy),
        })
    return bins


def expected_calibration_error(records: list[dict[str, Any]], n_bins: int = 10) -> float:
    bins = reliability_bins(records, n_bins=n_bins)
    total = sum(int(item["n"]) for item in bins)
    if total == 0:
        return math.nan
    return sum((int(item["n"]) / total) * float(item["gap"]) for item in bins if int(item["n"]) > 0)


def _nested_number(row: dict[str, Any], paths: list[tuple[str, ...]]) -> float | None:
    for path in paths:
        value: Any = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        number = _as_float(value)
        if number is not None:
            return number
    return None


def end_to_end_latency(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(_as_float(row.get("latency_ms")) for row in records)


def revision_latency(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        _nested_number(row, [
            ("metadata", "revision", "latency_ms"),
            ("metadata", "revision_latency_ms"),
            ("metadata", "native_execution", "revision_latency_ms"),
        ])
        for row in records
    )


def tokens(records: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for row in records:
        total = _nested_number(row, [("metadata", "total_tokens")])
        if total is not None:
            values.append(total)
            continue
        input_tokens = _as_float(row.get("input_tokens"))
        output_tokens = _as_float(row.get("output_tokens"))
        if input_tokens is not None and output_tokens is not None:
            values.append(input_tokens + output_tokens)
    return _mean_or_nan(values)


def model_calls(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        _nested_number(row, [
            ("metadata", "model_calls"),
            ("metadata", "native_execution", "model_calls"),
            ("metadata", "verification", "model_calls"),
            ("metadata", "revision", "model_calls"),
        ])
        for row in records
    )


def nli_calls(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        _nested_number(row, [
            ("metadata", "nli_calls"),
            ("metadata", "verification", "nli_calls"),
            ("metadata", "native_execution", "nli_calls"),
        ])
        for row in records
    )


def peak_memory(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        _nested_number(row, [
            ("metadata", "peak_memory_mb"),
            ("metadata", "memory", "peak_mb"),
            ("metadata", "native_execution", "peak_memory_mb"),
        ])
        for row in records
    )


def estimated_cost(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan(
        _nested_number(row, [
            ("metadata", "estimated_cost_usd"),
            ("metadata", "cost_usd"),
            ("metadata", "native_execution", "cost_usd"),
        ])
        for row in records
    )


SCALAR_METRICS: dict[str, Callable[[list[dict[str, Any]]], float]] = {
    "answer_text_accuracy": answer_text_accuracy,
    "end_to_end_answer_accuracy": end_to_end_answer_accuracy,
    "recency_flip_rate": recency_flip_rate,
    "probability_recency_sensitivity": probability_recency_sensitivity,
    "unsupported_supersession_rate": unsupported_supersession_rate,
    "order_reversal_consistency": order_reversal_consistency,
    "repetition_resistance": repetition_resistance,
    "stable_and_correct": stable_and_correct,
    "stable_but_wrong": stable_but_wrong,
    "flipped_to_correct": flipped_to_correct,
    "flipped_to_wrong": flipped_to_wrong,
    "genuine_update_accuracy": genuine_update_accuracy,
    "retroactive_correction_accuracy": retroactive_correction_accuracy,
    "historical_state_accuracy": historical_state_accuracy,
    "current_state_accuracy": current_state_accuracy,
    "stale_belief_rate": stale_belief_rate,
    "conflict_precision": conflict_precision,
    "conflict_recall": conflict_recall,
    "conflict_macro_f1": conflict_macro_f1,
    "conflict_f1": conflict_f1,
    "conflict_specificity": conflict_specificity,
    "false_conflict_rate": false_conflict_rate,
    "abstention_precision": abstention_precision,
    "abstention_recall": abstention_recall,
    "abstention_f1": abstention_f1,
    "selective_risk": selective_risk,
    "coverage": coverage,
    "area_under_risk_coverage_curve": area_under_risk_coverage_curve,
    "accuracy_at_fixed_coverage": accuracy_at_fixed_coverage,
    "citation_compliance_rate": citation_compliance_rate,
    "citation_support_rate": citation_support_rate,
    "citation_support_precision": citation_support_precision,
    "citation_support_recall": citation_support_recall,
    "citation_support_f1": citation_support_f1,
    "contradiction_f1": contradiction_f1,
    "neutral_insufficient_f1": neutral_insufficient_f1,
    "temporal_mismatch_f1": temporal_mismatch_f1,
    "supported_and_correct": supported_and_correct,
    "compliant_but_unsupported": compliant_but_unsupported,
    "supported_but_wrong": supported_but_wrong,
    "supported_and_text_correct": supported_and_text_correct,
    "supported_but_text_wrong": supported_but_text_wrong,
    "brier_score": brier_score,
    "expected_calibration_error": expected_calibration_error,
    "end_to_end_latency": end_to_end_latency,
    "revision_latency": revision_latency,
    "tokens": tokens,
    "model_calls": model_calls,
    "nli_calls": nli_calls,
    "peak_memory": peak_memory,
    "estimated_cost": estimated_cost,
    "answer_accuracy": answer_accuracy,
}


STRUCTURED_METRICS = {
    "conflict_versus_evolution_confusion_matrix": conflict_evolution_confusion_matrix,
    "reliability_bins": reliability_bins,
    "risk_coverage_curve": risk_coverage_curve,
}


METRICS = SCALAR_METRICS
