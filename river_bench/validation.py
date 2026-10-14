"""Validation helpers and CLI-facing checks for RIVER-Bench JSONL files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import BenchmarkExample


def load_examples(path: Path) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(BenchmarkExample.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                errors.append(f"{path}:{line_number}: {exc}")
    if errors:
        raise ValueError("\n".join(errors))
    return examples


def validate_unique_ids(examples: list[BenchmarkExample]) -> None:
    ids = [item.example_id for item in examples]
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate example_id values: {duplicates}")


def validate_pair_completeness(examples: list[BenchmarkExample]) -> None:
    grouped: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for item in examples:
        grouped[item.pair_id].append(item)

    failures: list[str] = []
    for pair_id, items in sorted(grouped.items()):
        if len(items) != 2:
            failures.append(f"{pair_id}: expected 2 variants, found {len(items)}")
            continue
        interventions = {item.intervention for item in items}
        base_ids = {item.base_example_id for item in items}
        splits = {item.split for item in items}
        if len(interventions) != 1:
            failures.append(f"{pair_id}: mixed interventions {sorted(interventions)}")
        if len(base_ids) != 1:
            failures.append(f"{pair_id}: mixed base_example_id values {sorted(base_ids)}")
        if len(splits) != 1:
            failures.append(f"{pair_id}: variants cross splits {sorted(splits)}")
    if failures:
        raise ValueError("\n".join(failures))


# Arrival order is exactly what `order_reversal` manipulates, so the paired rows
# are compared on evidence content while the arrival timestamps are allowed to
# permute. A separate check asserts they really are a permutation.
ORDER_PERMUTED_FIELDS = ("transaction_time",)

# Verifier scores are measured separately for each variant of a pair, so the
# same evidence text can differ in the sixth decimal place. That is numerical
# noise, not a change of evidence.
VERIFIER_SCORE_FIELDS = ("entailment", "contradiction", "neutral")
VERIFIER_SCORE_TOLERANCE = 1e-5


def _canonical_evidence_multiset(
    example: BenchmarkExample, *, ignore_fields: tuple[str, ...] = ()
) -> Counter[str]:
    serialized = []
    for evidence in example.evidence:
        payload = evidence.model_dump(mode="json")
        for field in ignore_fields:
            payload.pop(field, None)
        serialized.append(json.dumps(payload, sort_keys=True))
    return Counter(serialized)


def _same_evidence_up_to_score_noise(
    left: BenchmarkExample, right: BenchmarkExample, *, ignore_fields: tuple[str, ...] = ()
) -> bool:
    ignored = (*ignore_fields, *VERIFIER_SCORE_FIELDS)
    if _canonical_evidence_multiset(left, ignore_fields=ignored) != _canonical_evidence_multiset(
        right, ignore_fields=ignored
    ):
        return False
    by_id = {evidence.evidence_id: evidence for evidence in right.evidence}
    return all(
        evidence.evidence_id in by_id
        and all(
            abs(getattr(evidence, field) - getattr(by_id[evidence.evidence_id], field))
            <= VERIFIER_SCORE_TOLERANCE
            for field in VERIFIER_SCORE_FIELDS
        )
        for evidence in left.evidence
    )


def validate_order_only_equivalence(examples: list[BenchmarkExample]) -> None:
    grouped: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for item in examples:
        if item.intervention == "order_reversal":
            grouped[item.pair_id].append(item)

    failures: list[str] = []
    for pair_id, items in sorted(grouped.items()):
        if len(items) != 2:
            failures.append(f"{pair_id}: order_reversal pair is incomplete")
            continue
        left, right = sorted(items, key=lambda item: item.example_id)
        if not _same_evidence_up_to_score_noise(left, right, ignore_fields=ORDER_PERMUTED_FIELDS):
            failures.append(f"{pair_id}: evidence multiset changed under order reversal")
        if Counter(item.transaction_time for item in left.evidence) != Counter(
            item.transaction_time for item in right.evidence
        ):
            failures.append(f"{pair_id}: arrival timestamps are not a permutation under order reversal")
        if len(left.hypotheses) < 2:
            failures.append(
                f"{pair_id}: order_reversal needs >= 2 candidate hypotheses or no system can flip"
            )
        if left.query != right.query:
            failures.append(f"{pair_id}: query changed under order reversal")
        if left.target_time != right.target_time:
            failures.append(f"{pair_id}: target_time changed under order reversal")
        if left.gold_state != right.gold_state or left.gold_hypothesis_id != right.gold_hypothesis_id:
            failures.append(f"{pair_id}: gold label changed under order reversal")
    if failures:
        raise ValueError("\n".join(failures))


def _leakage_keys(example: BenchmarkExample) -> set[str]:
    metadata = example.transformation_metadata
    keys = set(metadata.get("leakage_keys", []))
    entity = metadata.get("entity")
    event_id = metadata.get("event_id")
    if entity:
        keys.add(f"entity:{entity}")
    if event_id:
        keys.add(f"event:{event_id}")
    for evidence in example.evidence:
        keys.add(f"entity:{evidence.subject}")
    return keys


def validate_split_leakage(examples: list[BenchmarkExample]) -> None:
    by_base: dict[str, set[str]] = defaultdict(set)
    by_key: dict[str, set[str]] = defaultdict(set)

    for example in examples:
        by_base[example.base_example_id].add(example.split)
        for key in _leakage_keys(example):
            by_key[key].add(example.split)

    failures = [
        f"{base}: base example appears in splits {sorted(splits)}"
        for base, splits in sorted(by_base.items())
        if len(splits) > 1
    ]
    failures.extend(
        f"{key}: leakage key appears in splits {sorted(splits)}"
        for key, splits in sorted(by_key.items())
        if len(splits) > 1
    )
    if failures:
        raise ValueError("\n".join(failures))


def intervention_counts(examples: list[BenchmarkExample]) -> dict[str, int]:
    return dict(sorted(Counter(item.intervention for item in examples).items()))


def validate_all(examples: list[BenchmarkExample]) -> None:
    validate_unique_ids(examples)
    validate_pair_completeness(examples)
    validate_order_only_equivalence(examples)
    validate_split_leakage(examples)


def examples_to_jsonl(examples: list[BenchmarkExample]) -> str:
    return "\n".join(item.model_dump_json() for item in examples) + "\n"


def examples_as_runner_rows(examples: list[BenchmarkExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        row = example.model_dump(mode="json")
        row["evidence"] = example.evidence_for_runner()
        row["hypotheses"] = example.hypotheses_for_runner()
        rows.append(row)
    return rows
