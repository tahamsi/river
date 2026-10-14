from pathlib import Path

import pytest

from river.revision import RIVERDeterministic
from river.schemas import EvidenceRecord, RevisionState
from river_bench.transformations import build_authored_examples, load_seed_manifest
from river_bench.validation import (
    examples_to_jsonl,
    load_examples,
    validate_order_only_equivalence,
    validate_pair_completeness,
    validate_split_leakage,
    validate_unique_ids,
)


FIXTURE = Path("river_bench/processed/river_bench_unit.jsonl")
SEEDS = Path("river_bench/manifests/river_bench_seed_fixtures.jsonl")


def _examples():
    return load_examples(FIXTURE)


def _pair(examples, intervention):
    return sorted(
        [item for item in examples if item.intervention == intervention],
        key=lambda item: item.example_id,
    )[:2]


def _decision(example):
    evidence = [EvidenceRecord.model_validate(item.as_river_record()) for item in example.evidence]
    return RIVERDeterministic().resolve(example.hypotheses, evidence, example.target_time)


def test_authored_transformations_are_deterministic_and_fixture_is_current():
    seeds = load_seed_manifest(SEEDS)
    first = build_authored_examples(seeds)
    second = build_authored_examples(seeds)

    assert [item.model_dump(mode="json") for item in first] == [
        item.model_dump(mode="json") for item in second
    ]
    assert examples_to_jsonl(first) == FIXTURE.read_text(encoding="utf-8")


def test_unique_ids():
    validate_unique_ids(_examples())


def test_pair_integrity():
    validate_pair_completeness(_examples())


def test_split_integrity():
    validate_split_leakage(_examples())


def test_order_reversal_preserves_evidence_multiset():
    examples = _examples()
    validate_order_only_equivalence(examples)
    left, right = _pair(examples, "order_reversal")
    assert [item.evidence_id for item in left.evidence] == list(
        reversed([item.evidence_id for item in right.evidence])
    )


def _with_shifted_entailment(examples, delta):
    left, right = _pair(examples, "order_reversal")
    shifted = right.evidence[0].model_copy(
        update={"entailment": min(1.0, right.evidence[0].entailment + delta)}
    )
    right = right.model_copy(update={"evidence": [shifted, *right.evidence[1:]]})
    return [left, right]


def test_order_reversal_tolerates_verifier_score_noise():
    validate_order_only_equivalence(_with_shifted_entailment(_examples(), 1e-6))


def test_order_reversal_rejects_a_changed_verifier_score():
    with pytest.raises(ValueError, match="evidence multiset changed"):
        validate_order_only_equivalence(_with_shifted_entailment(_examples(), 0.01))


def test_supported_update_changes_gold_hypothesis_and_river_selects_update():
    control, variant = _pair(_examples(), "supported_genuine_update")

    assert control.gold_hypothesis_id != variant.gold_hypothesis_id
    assert variant.gold_hypothesis_id == variant.intervention_hypothesis_id

    decision = _decision(variant)
    assert decision.state == RevisionState.ACTIVE
    assert decision.selected_hypothesis_id == variant.gold_hypothesis_id


def test_unsupported_recency_does_not_change_gold_hypothesis():
    examples = [
        item
        for item in _examples()
        if item.intervention == "unsupported_late_assertion"
        and item.transformation_metadata["variant_role"] == "late_unsupported_assertion"
    ]

    assert examples
    for example in examples:
        assert example.gold_hypothesis_id != example.intervention_hypothesis_id
        decision = _decision(example)
        assert decision.state == RevisionState.ACTIVE
        assert decision.selected_hypothesis_id == example.gold_hypothesis_id


def test_temporal_evolution_produces_disjoint_validity_intervals():
    example = _pair(_examples(), "temporal_evolution")[0]
    left, right = example.evidence

    assert not left.valid_time.overlaps(right.valid_time)
    decision = _decision(example)
    assert decision.state == RevisionState.EVOLUTION


def test_concurrent_conflict_produces_overlapping_validity_intervals():
    example = _pair(_examples(), "concurrent_disagreement")[0]
    left, right = example.evidence

    assert left.valid_time.overlaps(right.valid_time)
    decision = _decision(example)
    assert decision.state == RevisionState.DISPUTED
