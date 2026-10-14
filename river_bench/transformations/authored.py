"""Deterministic transformations from small authored seed fixtures."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from river.schemas import BeliefHypothesis, RevisionState

from river_bench.schemas import BenchmarkEvidence, BenchmarkExample


# Entailment for a chunk that is retrieved and on-topic but decides nothing.
RESIDUAL_ENTAILMENT = 0.12

INTERVENTIONS = {
    "order_reversal",
    "unsupported_late_assertion",
    "repeated_weak_assertion",
    "supported_genuine_update",
    "retroactive_correction",
    "temporal_evolution",
    "concurrent_disagreement",
    "evidence_removal",
    "insufficient_evidence_nei",
}


def load_seed_manifest(path: Path) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            intervention = item.get("intervention")
            if intervention not in INTERVENTIONS:
                raise ValueError(f"{path}:{line_number}: unknown intervention {intervention!r}")
            seeds.append(item)
    return seeds


def _iso(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.isoformat().replace("+00:00", "Z")


def _hypothesis(seed: dict[str, Any], suffix: str, obj: str, evidence_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "hypothesis_id": f"{seed['base_example_id']}:{suffix}",
        "subject": seed["subject"],
        "predicate": seed["predicate"],
        "object": obj,
        "evidence_ids": evidence_ids or [],
    }


def _evidence(
    seed: dict[str, Any],
    suffix: str,
    hypothesis: dict[str, Any],
    text: str,
    transaction_time: str,
    valid_start: str | None,
    valid_end: str | None,
    entailment: float,
    *,
    contradiction: float = 0.0,
    neutral: float | None = None,
    provenance_reliability: float = 1.0,
    independence_group: str | None = None,
) -> dict[str, Any]:
    evidence_id = f"{seed['base_example_id']}:{suffix}"
    if neutral is None:
        neutral = max(0.0, 1.0 - entailment - contradiction)
    return {
        "evidence_id": evidence_id,
        "source_id": f"{seed['source_dataset']}:{seed['base_example_id']}:{suffix}",
        "chunk_id": f"{seed['base_example_id']}:{suffix}:chunk0",
        "text": text,
        "subject": seed["subject"],
        "predicate": seed["predicate"],
        "object": hypothesis["object"],
        "hypothesis_id": hypothesis["hypothesis_id"],
        "transaction_time": _iso(transaction_time),
        "valid_time": {
            "start": _iso(valid_start),
            "end": _iso(valid_end),
            "confidence": 1.0,
        },
        "entailment": entailment,
        "contradiction": contradiction,
        "neutral": neutral,
        "extraction_confidence": 1.0,
        "provenance_reliability": provenance_reliability,
        "independence_group": independence_group or evidence_id,
    }


def _attach_evidence(hypotheses: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hypothesis: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        by_hypothesis[item["hypothesis_id"]].append(item["evidence_id"])
    attached: list[dict[str, Any]] = []
    for item in hypotheses:
        clone = dict(item)
        clone["evidence_ids"] = sorted(by_hypothesis.get(item["hypothesis_id"], []))
        attached.append(clone)
    return attached


def _example(
    seed: dict[str, Any],
    suffix: str,
    variant_role: str,
    evidence: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    *,
    target_time: str | None,
    gold_state: str,
    gold_hypothesis_id: str | None,
    intervention_hypothesis_id: str | None = None,
    query: str | None = None,
) -> BenchmarkExample:
    attached = _attach_evidence(hypotheses, evidence)
    metadata = {
        "generator": "authored_seed_transformations_v0",
        "variant_role": variant_role,
        "seed_base_example_id": seed["base_example_id"],
        "entity": seed["subject"],
        "event_id": seed.get("event_id", seed["base_example_id"]),
        "leakage_keys": [
            f"entity:{seed['subject']}",
            f"event:{seed.get('event_id', seed['base_example_id'])}",
        ],
    }
    metadata.update(seed.get("metadata", {}))
    return BenchmarkExample.model_validate(
        {
            "example_id": f"{seed['base_example_id']}:{suffix}",
            "base_example_id": seed["base_example_id"],
            "pair_id": f"{seed['base_example_id']}:{seed['intervention']}",
            "dataset": "river_bench_v0",
            "source_dataset": seed["source_dataset"],
            "split": seed["split"],
            "intervention": seed["intervention"],
            "session_id": f"{seed['base_example_id']}:{variant_role}",
            "query": query or seed["query"],
            "target_time": _iso(target_time),
            "evidence": [BenchmarkEvidence.model_validate(item).model_dump(mode="json") for item in evidence],
            "hypotheses": [BeliefHypothesis.model_validate(item).model_dump(mode="json") for item in attached],
            "gold_state": RevisionState(gold_state),
            "gold_hypothesis_id": gold_hypothesis_id,
            "intervention_hypothesis_id": intervention_hypothesis_id,
            "transformation_metadata": metadata,
        }
    )


def _order_reversal(seed: dict[str, Any]) -> list[BenchmarkExample]:
    h_gold = _hypothesis(seed, "h_supported", seed["object_a"])
    h_alt = _hypothesis(seed, "h_late", seed["object_b"])
    evidence = [
        _evidence(
            seed,
            "e_supported",
            h_gold,
            seed["support_text"],
            seed["t1"],
            seed["valid_start"],
            seed.get("valid_end"),
            0.92,
        ),
        _evidence(
            seed,
            "e_weak_alt",
            h_alt,
            seed["intervention_text"],
            seed["t2"],
            seed["valid_start"],
            seed.get("valid_end"),
            0.24,
        ),
    ]
    hypotheses = [h_gold, h_alt]
    return [
        _example(
            seed,
            "a",
            "original_order",
            evidence,
            hypotheses,
            target_time=seed["target_time"],
            gold_state="active",
            gold_hypothesis_id=h_gold["hypothesis_id"],
            intervention_hypothesis_id=h_alt["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "reversed_order",
            list(reversed(evidence)),
            hypotheses,
            target_time=seed["target_time"],
            gold_state="active",
            gold_hypothesis_id=h_gold["hypothesis_id"],
            intervention_hypothesis_id=h_alt["hypothesis_id"],
        ),
    ]


def _unsupported_late_assertion(seed: dict[str, Any]) -> list[BenchmarkExample]:
    h_gold = _hypothesis(seed, "h_supported", seed["object_a"])
    h_late = _hypothesis(seed, "h_unsupported_late", seed["object_b"])
    support = _evidence(
        seed,
        "e_supported",
        h_gold,
        seed["support_text"],
        seed["t1"],
        seed["valid_start"],
        seed.get("valid_end"),
        0.93,
    )
    unsupported = _evidence(
        seed,
        "e_late_unsupported",
        h_late,
        seed["intervention_text"],
        seed["t2"],
        seed["valid_start"],
        seed.get("valid_end"),
        0.21,
        independence_group=f"{seed['base_example_id']}:unsupported_repeat",
    )
    hypotheses = [h_gold, h_late]
    return [
        _example(
            seed,
            "a",
            "supported_control",
            [support],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="active",
            gold_hypothesis_id=h_gold["hypothesis_id"],
            intervention_hypothesis_id=h_late["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "late_unsupported_assertion",
            [support, unsupported],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="active",
            gold_hypothesis_id=h_gold["hypothesis_id"],
            intervention_hypothesis_id=h_late["hypothesis_id"],
        ),
    ]


def _repeated_weak_assertion(seed: dict[str, Any]) -> list[BenchmarkExample]:
    h_gold = _hypothesis(seed, "h_supported", seed["object_a"])
    h_repeated = _hypothesis(seed, "h_repeated_weak", seed["object_b"])
    support = _evidence(
        seed,
        "e_supported",
        h_gold,
        seed["support_text"],
        seed["t1"],
        seed["valid_start"],
        seed.get("valid_end"),
        0.89,
    )
    repeated = [
        _evidence(
            seed,
            f"e_weak_repeat_{index}",
            h_repeated,
            f"{seed['intervention_text']} Weak repost {index}.",
            seed[f"t{index + 1}"],
            seed["valid_start"],
            seed.get("valid_end"),
            0.20,
            independence_group=f"{seed['base_example_id']}:same_weak_channel",
        )
        for index in range(1, 4)
    ]
    hypotheses = [h_gold, h_repeated]
    return [
        _example(
            seed,
            "a",
            "supported_control",
            [support],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="active",
            gold_hypothesis_id=h_gold["hypothesis_id"],
            intervention_hypothesis_id=h_repeated["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "repeated_weak_assertions",
            [support, *repeated],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="active",
            gold_hypothesis_id=h_gold["hypothesis_id"],
            intervention_hypothesis_id=h_repeated["hypothesis_id"],
        ),
    ]


def _supported_genuine_update(seed: dict[str, Any]) -> list[BenchmarkExample]:
    h_old = _hypothesis(seed, "h_old", seed["object_a"])
    h_new = _hypothesis(seed, "h_new", seed["object_b"])
    old_control = _evidence(
        seed,
        "e_old_control",
        h_old,
        seed["support_text"],
        seed["t1"],
        seed["old_valid_start"],
        None,
        0.91,
    )
    old_superseded = _evidence(
        seed,
        "e_old_superseded",
        h_old,
        seed["support_text"],
        seed["t1"],
        seed["old_valid_start"],
        seed["old_valid_end"],
        0.91,
    )
    new_supported = _evidence(
        seed,
        "e_new_supported",
        h_new,
        seed["intervention_text"],
        seed["t2"],
        seed["new_valid_start"],
        None,
        0.94,
    )
    hypotheses = [h_old, h_new]
    return [
        _example(
            seed,
            "a",
            "pre_update_control",
            [old_control],
            hypotheses,
            target_time=seed["control_target_time"],
            gold_state="active",
            gold_hypothesis_id=h_old["hypothesis_id"],
            intervention_hypothesis_id=h_new["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "supported_update_variant",
            [old_superseded, new_supported],
            hypotheses,
            target_time=seed["variant_target_time"],
            gold_state="active",
            gold_hypothesis_id=h_new["hypothesis_id"],
            intervention_hypothesis_id=h_new["hypothesis_id"],
        ),
    ]


def _retroactive_correction(seed: dict[str, Any]) -> list[BenchmarkExample]:
    h_original = _hypothesis(seed, "h_original", seed["object_a"])
    h_corrected = _hypothesis(seed, "h_corrected", seed["object_b"])
    original = _evidence(
        seed,
        "e_original",
        h_original,
        seed["support_text"],
        seed["t1"],
        seed["valid_start"],
        None,
        0.74,
    )
    correction = _evidence(
        seed,
        "e_late_correction",
        h_corrected,
        seed["intervention_text"],
        seed["t2"],
        seed["valid_start"],
        None,
        0.96,
    )
    hypotheses = [h_original, h_corrected]
    return [
        _example(
            seed,
            "a",
            "original_record",
            [original],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="active",
            gold_hypothesis_id=h_original["hypothesis_id"],
            intervention_hypothesis_id=h_corrected["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "retroactive_correction",
            [original, correction],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="active",
            gold_hypothesis_id=h_corrected["hypothesis_id"],
            intervention_hypothesis_id=h_corrected["hypothesis_id"],
        ),
    ]


def _temporal_evolution(seed: dict[str, Any]) -> list[BenchmarkExample]:
    h_phase_one = _hypothesis(seed, "h_phase_one", seed["object_a"])
    h_phase_two = _hypothesis(seed, "h_phase_two", seed["object_b"])
    phase_one = _evidence(
        seed,
        "e_phase_one",
        h_phase_one,
        seed["support_text"],
        seed["t1"],
        seed["old_valid_start"],
        seed["old_valid_end"],
        0.90,
    )
    phase_two = _evidence(
        seed,
        "e_phase_two",
        h_phase_two,
        seed["intervention_text"],
        seed["t2"],
        seed["new_valid_start"],
        seed.get("new_valid_end"),
        0.91,
    )
    hypotheses = [h_phase_one, h_phase_two]
    return [
        _example(
            seed,
            "a",
            "chronological_validity",
            [phase_one, phase_two],
            hypotheses,
            target_time=None,
            gold_state="evolution",
            gold_hypothesis_id=None,
            intervention_hypothesis_id=h_phase_two["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "reversed_transaction_order",
            [phase_two, phase_one],
            hypotheses,
            target_time=None,
            gold_state="evolution",
            gold_hypothesis_id=None,
            intervention_hypothesis_id=h_phase_two["hypothesis_id"],
        ),
    ]


def _concurrent_disagreement(seed: dict[str, Any]) -> list[BenchmarkExample]:
    h_left = _hypothesis(seed, "h_left", seed["object_a"])
    h_right = _hypothesis(seed, "h_right", seed["object_b"])
    left = _evidence(
        seed,
        "e_left",
        h_left,
        seed["support_text"],
        seed["t1"],
        seed["valid_start"],
        seed["valid_end"],
        0.86,
    )
    right = _evidence(
        seed,
        "e_right",
        h_right,
        seed["intervention_text"],
        seed["t2"],
        seed["overlap_start"],
        seed["overlap_end"],
        0.86,
    )
    hypotheses = [h_left, h_right]
    return [
        _example(
            seed,
            "a",
            "source_a_first",
            [left, right],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="disputed",
            gold_hypothesis_id=None,
            intervention_hypothesis_id=h_right["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "source_b_first",
            [right, left],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="disputed",
            gold_hypothesis_id=None,
            intervention_hypothesis_id=h_right["hypothesis_id"],
        ),
    ]


def _evidence_removal(seed: dict[str, Any]) -> list[BenchmarkExample]:
    h_supported = _hypothesis(seed, "h_supported", seed["object_a"])
    support = _evidence(
        seed,
        "e_supported",
        h_supported,
        seed["support_text"],
        seed["t1"],
        seed["valid_start"],
        seed.get("valid_end"),
        0.90,
    )
    # The removal variant drops the decisive chunk but retrieval still returns
    # something on-topic. Leaving the bundle empty would make evidence count a
    # perfect predictor of the abstain label.
    residual = _evidence(
        seed,
        "e_residual",
        h_supported,
        f"This passage concerns {seed['subject']} but does not state the requested value.",
        seed["t1"],
        None,
        None,
        RESIDUAL_ENTAILMENT,
        neutral=1.0 - RESIDUAL_ENTAILMENT,
    )
    hypotheses = [h_supported]
    return [
        _example(
            seed,
            "a",
            "evidence_present",
            [support, residual],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="active",
            gold_hypothesis_id=h_supported["hypothesis_id"],
            intervention_hypothesis_id=None,
        ),
        _example(
            seed,
            "b",
            "evidence_removed",
            [residual],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="abstain",
            gold_hypothesis_id=None,
            intervention_hypothesis_id=h_supported["hypothesis_id"],
        ),
    ]


def _insufficient_evidence_nei(seed: dict[str, Any]) -> list[BenchmarkExample]:
    h_claim = _hypothesis(seed, "h_claim", seed["object_a"])
    neutral_context = _evidence(
        seed,
        "e_neutral_context",
        h_claim,
        seed["support_text"],
        seed["t1"],
        seed["valid_start"],
        seed.get("valid_end"),
        0.05,
        neutral=0.91,
    )
    hypotheses = [h_claim]
    return [
        _example(
            seed,
            "a",
            "no_evidence",
            [],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="abstain",
            gold_hypothesis_id=None,
            intervention_hypothesis_id=h_claim["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "neutral_evidence_only",
            [neutral_context],
            hypotheses,
            target_time=seed["target_time"],
            gold_state="abstain",
            gold_hypothesis_id=None,
            intervention_hypothesis_id=h_claim["hypothesis_id"],
        ),
    ]


BUILDERS = {
    "order_reversal": _order_reversal,
    "unsupported_late_assertion": _unsupported_late_assertion,
    "repeated_weak_assertion": _repeated_weak_assertion,
    "supported_genuine_update": _supported_genuine_update,
    "retroactive_correction": _retroactive_correction,
    "temporal_evolution": _temporal_evolution,
    "concurrent_disagreement": _concurrent_disagreement,
    "evidence_removal": _evidence_removal,
    "insufficient_evidence_nei": _insufficient_evidence_nei,
}


def build_authored_examples(seeds: list[dict[str, Any]]) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for seed in seeds:
        examples.extend(BUILDERS[seed["intervention"]](seed))
    return examples
