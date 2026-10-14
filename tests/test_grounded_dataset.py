"""Structural invariants for the grounded corpus built from source records.

The rest of the suite uses the 20-row authored fixture. These tests read the
generated train and validation splits so that a task family cannot silently
become unmeasurable.

Skipped when ``data/grounded/`` is absent; that corpus is not part of the public
release.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pytest

GROUNDED = Path(os.environ.get("RIVER_GROUNDED_DIR", "data/grounded"))
SPLITS = ("train", "dev", "test")

pytestmark = pytest.mark.skipif(
    not all((GROUNDED / f"river_bench_{name}.jsonl").exists() for name in SPLITS),
    reason="three-way grounded splits not present; run scripts.benchmark.build_grounded_river_bench",
)


def _load(split: str) -> list[dict]:
    path = GROUNDED / f"river_bench_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [row for split in SPLITS for row in _load(split)]


def test_order_reversal_offers_an_alternative_to_flip_to(rows):
    """A single-hypothesis order pair makes the recency flip rate identically zero."""

    offenders = [
        row["example_id"]
        for row in rows
        if row["intervention"] == "order_reversal" and len(row["hypotheses"]) < 2
    ]
    assert not offenders, f"{len(offenders)} order_reversal rows cannot flip: {offenders[:3]}"


def test_order_reversal_permutes_arrival_time(rows):
    """Reversal must move the newest arrival, not only the list position."""

    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["intervention"] == "order_reversal":
            pairs[row["pair_id"]].append(row)

    assert pairs, "no order_reversal pairs found"
    for pair_id, variants in pairs.items():
        assert len(variants) == 2, f"{pair_id}: incomplete pair"
        left, right = sorted(variants, key=lambda row: row["example_id"])
        assert Counter(item["transaction_time"] for item in left["evidence"]) == Counter(
            item["transaction_time"] for item in right["evidence"]
        ), f"{pair_id}: arrival stamps are not a permutation"

        def newest(row: dict) -> str:
            return max(row["evidence"], key=lambda item: item["transaction_time"])["evidence_id"]

        assert newest(left) != newest(right), f"{pair_id}: the same evidence arrives last in both variants"


def test_gold_state_is_not_predictable_from_bundle_shape(rows):
    """The leakage canary.

    If evidence count or hypothesis count determines ``gold_state``, a system can
    score the benchmark without reading any evidence, and every abstention metric
    becomes meaningless.
    """

    for feature, name in ((lambda row: len(row["evidence"]), "evidence count"),
                          (lambda row: len(row["hypotheses"]), "hypothesis count")):
        by_value: dict[int, set[str]] = defaultdict(set)
        for row in rows:
            by_value[feature(row)].add(row["gold_state"])
        determined = {value: states for value, states in by_value.items() if len(states) == 1}
        total = sum(1 for row in rows if feature(row) in determined)
        assert total / len(rows) < 0.5, (
            f"{name} determines gold_state for {total}/{len(rows)} rows: {determined}"
        )


def test_no_intervention_family_has_an_empty_bundle(rows):
    """An empty bundle short-circuits revision before any scoring happens."""

    empty = Counter(row["intervention"] for row in rows if not row["evidence"])
    assert not empty, f"intervention families present no evidence at all: {dict(empty)}"


def test_valid_time_coverage_where_the_mechanism_is_under_test(rows):
    """Valid time is RIVER's core signal; families that test it must populate it."""

    required = {"temporal_evolution", "supported_genuine_update", "retroactive_correction"}
    for intervention in sorted(required):
        items = [item for row in rows if row["intervention"] == intervention for item in row["evidence"]]
        assert items, f"{intervention}: no evidence"
        dated = [
            item
            for item in items
            if (item.get("valid_time") or {}).get("start") or (item.get("valid_time") or {}).get("end")
        ]
        assert len(dated) == len(items), (
            f"{intervention}: {len(items) - len(dated)}/{len(items)} evidence items have no valid_time"
        )


def test_pairs_never_cross_the_split_boundary():
    """Leakage check on the shipped splits rather than a regenerated fixture."""

    by_split = {split: {row["pair_id"] for row in _load(split)} for split in SPLITS}
    overlap = set().union(*(by_split[a] & by_split[b] for a in SPLITS for b in SPLITS if a < b))
    assert not overlap, f"{len(overlap)} pair_ids appear in both splits: {sorted(overlap)[:3]}"


def test_declared_review_status_is_explicit(rows):
    """Every row states whether a human approved its labels."""

    statuses = Counter(row["transformation_metadata"]["human_review_status"] for row in rows)
    assert set(statuses) <= {"accepted", "pending"}, f"unexpected review statuses: {dict(statuses)}"


def test_gold_is_not_identifiable_from_answer_length(rows):
    """The gold answer must not be separable from its alternative by shape.

    An earlier build gave the order_reversal alternative the whole distractor
    sentence as its object while the gold answer was a short entity, so "pick
    the shorter string" scored perfectly on the family that carries the recency
    flip rate. Chance is 50%; a family well above that is encoding the answer
    in a place no reader of the evidence would look.
    """

    tally: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        hypotheses = row.get("hypotheses") or []
        gold = row.get("gold_hypothesis_id")
        if len(hypotheses) < 2 or gold is None:
            continue
        objects = {item["hypothesis_id"]: str(item.get("object", "")) for item in hypotheses}
        if gold not in objects:
            continue
        shortest = min(objects, key=lambda key: len(objects[key]))
        tally[row["intervention"]].append(shortest == gold)

    assert tally, "no multi-hypothesis rows found"
    offenders = {
        family: round(sum(hits) / len(hits), 3)
        for family, hits in tally.items()
        if not 0.25 < sum(hits) / len(hits) < 0.75
    }
    assert not offenders, f"gold is predictable from answer length: {offenders}"
