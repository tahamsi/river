import math

import pytest

from river.evaluation.metrics import (
    EvaluationError,
    abstention_f1,
    abstention_precision,
    abstention_recall,
    accuracy_at_fixed_coverage,
    area_under_risk_coverage_curve,
    brier_score,
    citation_support_f1,
    citation_support_precision,
    citation_support_recall,
    conflict_evolution_confusion_matrix,
    conflict_macro_f1,
    conflict_precision,
    conflict_recall,
    conflict_specificity,
    current_state_accuracy,
    end_to_end_latency,
    expected_calibration_error,
    false_conflict_rate,
    flipped_to_correct,
    flipped_to_wrong,
    genuine_update_accuracy,
    historical_state_accuracy,
    neutral_insufficient_f1,
    order_reversal_consistency,
    probability_recency_sensitivity,
    recency_flip_rate,
    repetition_resistance,
    retroactive_correction_accuracy,
    revision_latency,
    stable_and_correct,
    stable_but_wrong,
    stale_belief_rate,
    supported_and_correct,
    tokens,
    unsupported_supersession_rate,
)


def row(example_id, **overrides):
    payload = {
        "run_id": "run",
        "example_id": example_id,
        "base_example_id": example_id.split(":")[0],
        "pair_id": None,
        "dataset": "river_bench",
        "source_dataset": "unit",
        "split": "test",
        "intervention": "none",
        "comparison_mode": "controlled",
        "baseline": "river_d",
        "seed": 1,
        "gold_state": "active",
        "gold_hypothesis_id": "h1",
        "predicted_state": "active",
        "predicted_hypothesis_id": "h1",
        "configuration_hash": "sha256:config",
        "code_commit": "commit",
        "dataset_hash": "sha256:data",
        "prompt_hash": "sha256:prompt",
    }
    payload.update(overrides)
    return payload


def test_recency_metrics_use_complete_pairs_and_probabilities():
    rows = [
        row("p1:a", pair_id="p1", intervention="order_reversal", state_probabilities={"active": 0.8, "abstain": 0.2}),
        row("p1:b", pair_id="p1", intervention="order_reversal", state_probabilities={"active": 0.8, "abstain": 0.2}),
        row("p2:a", pair_id="p2", intervention="order_reversal", predicted_hypothesis_id="h1", state_probabilities={"active": 1.0, "abstain": 0.0}),
        row("p2:b", pair_id="p2", intervention="order_reversal", predicted_hypothesis_id="h2", state_probabilities={"active": 0.0, "abstain": 1.0}),
    ]

    assert recency_flip_rate(rows) == 0.5
    assert order_reversal_consistency(rows) == 0.5
    assert probability_recency_sensitivity(rows) == pytest.approx(0.5)
    assert stable_and_correct(rows) == 0.5
    assert stable_but_wrong(rows) == 0.0
    assert flipped_to_correct(rows) == 0.0
    assert flipped_to_wrong(rows) == 0.5


def test_paired_metrics_require_pair_id():
    rows = [row("missing", intervention="order_reversal", pair_id=None)]

    with pytest.raises(EvaluationError, match="requires pair_id"):
        recency_flip_rate(rows)


def test_unsupported_supersession_and_repetition_resistance():
    rows = [
        row(
            "u1",
            intervention="unsupported_late_assertion",
            predicted_hypothesis_id="h_bad",
            intervention_hypothesis_id="h_bad",
            transformation_metadata={"variant_role": "late_unsupported_assertion"},
        ),
        row(
            "u2",
            intervention="unsupported_late_assertion",
            predicted_hypothesis_id="h_good",
            intervention_hypothesis_id="h_bad",
            transformation_metadata={"variant_role": "late_unsupported_assertion"},
        ),
        row(
            "r1",
            intervention="repeated_weak_assertion",
            predicted_hypothesis_id="h_bad",
            intervention_hypothesis_id="h_bad",
            transformation_metadata={"variant_role": "repeated_weak_assertions"},
        ),
        row(
            "r2",
            intervention="repeated_weak_assertion",
            predicted_hypothesis_id="h_good",
            intervention_hypothesis_id="h_bad",
            transformation_metadata={"variant_role": "repeated_weak_assertions"},
        ),
    ]

    assert unsupported_supersession_rate(rows) == 0.5
    assert repetition_resistance(rows) == 0.5


def test_update_and_temporal_metrics_are_hand_calculated():
    rows = [
        row("g1", intervention="supported_genuine_update", gold_hypothesis_id="h_new", intervention_hypothesis_id="h_new", predicted_hypothesis_id="h_new"),
        row("g2", intervention="supported_genuine_update", gold_hypothesis_id="h_new", intervention_hypothesis_id="h_new", predicted_hypothesis_id="h_old"),
        row("retro", intervention="retroactive_correction", gold_hypothesis_id="h_fixed", intervention_hypothesis_id="h_fixed", predicted_hypothesis_id="h_fixed"),
        row("hist", transformation_metadata={"temporal_slice": "historical"}, predicted_hypothesis_id="h1"),
        row("current", transformation_metadata={"temporal_slice": "current"}, predicted_hypothesis_id="h2"),
        row("stale", transformation_metadata={"stale_hypothesis_id": "h_old"}, predicted_hypothesis_id="h_old"),
        row("fresh", transformation_metadata={"stale_hypothesis_id": "h_old"}, predicted_hypothesis_id="h_new"),
    ]

    assert genuine_update_accuracy(rows) == 0.5
    assert retroactive_correction_accuracy(rows) == 1.0
    assert historical_state_accuracy(rows) == 1.0
    assert current_state_accuracy(rows) == 0.0
    assert stale_belief_rate(rows) == 0.5


def test_conflict_and_abstention_metrics_are_hand_calculated():
    rows = [
        row("tp", gold_state="disputed", predicted_state="disputed", gold_hypothesis_id=None),
        row("fp", gold_state="active", predicted_state="disputed"),
        row("fn", gold_state="disputed", predicted_state="active", gold_hypothesis_id=None),
        row("tn", gold_state="active", predicted_state="active"),
        row("evo", gold_state="evolution", predicted_state="disputed", gold_hypothesis_id=None),
    ]

    assert conflict_precision(rows) == pytest.approx(1 / 3)
    assert conflict_recall(rows) == pytest.approx(0.5)
    assert conflict_specificity(rows) == pytest.approx(1 / 3)
    assert false_conflict_rate(rows) == pytest.approx(2 / 3)
    assert conflict_macro_f1(rows) == pytest.approx((0.4 + 0.4) / 2)
    matrix = conflict_evolution_confusion_matrix(rows)
    assert matrix["disputed"]["disputed"] == 1
    assert matrix["evolution"]["disputed"] == 1

    abstain_rows = [
        row("a_tp", gold_state="abstain", predicted_state="abstain", gold_hypothesis_id=None),
        row("a_fp", gold_state="active", predicted_state="abstain"),
        row("a_fn", gold_state="abstain", predicted_state="active", gold_hypothesis_id=None),
    ]
    assert abstention_precision(abstain_rows) == 0.5
    assert abstention_recall(abstain_rows) == 0.5
    assert abstention_f1(abstain_rows) == 0.5


def test_citation_metrics_require_labelled_claim_reference():
    rows = [
        row("c1", answer_correct=True, citation_supported=True, claim_verification_results=[
            {"claim_id": "c1", "claim_text": "A", "label": "supported", "citations": [{"source_id": "s", "chunk_id": "c"}], "metadata": {"gold_label": "supported"}},
        ]),
        row("c2", answer_correct=True, citation_supported=False, claim_verification_results=[
            {"claim_id": "c2", "claim_text": "B", "label": "supported", "citations": [{"source_id": "s", "chunk_id": "c"}], "metadata": {"gold_label": "neutral"}},
        ]),
        row("c3", answer_correct=False, citation_supported=True, claim_verification_results=[
            {"claim_id": "c3", "claim_text": "C", "label": "neutral", "citations": [{"source_id": "s", "chunk_id": "c"}], "metadata": {"gold_label": "supported"}},
        ]),
        row("c4", answer_correct=True, citation_supported=False, claim_verification_results=[
            {"claim_id": "c4", "claim_text": "D", "label": "neutral", "citations": [{"source_id": "s", "chunk_id": "c"}], "metadata": {"gold_label": "neutral"}},
        ]),
    ]

    assert citation_support_precision(rows) == 0.5
    assert citation_support_recall(rows) == 0.5
    assert citation_support_f1(rows) == 0.5
    assert neutral_insufficient_f1(rows) == 0.5
    assert supported_and_correct(rows) == pytest.approx(1 / 4)

    assert math.isnan(citation_support_f1([row("missing", claim_verification_results=[
        {"claim_id": "c", "claim_text": "A", "label": "supported", "citations": []}
    ])]))


def test_calibration_abstention_risk_and_efficiency_metrics():
    rows = [
        row("cal1", answer_correct=True, confidence=0.8, predicted_state="active", latency_ms=100, input_tokens=10, output_tokens=5, metadata={"revision": {"latency_ms": 20}}),
        row("cal2", answer_correct=False, confidence=0.2, predicted_state="active", latency_ms=300, input_tokens=20, output_tokens=10, metadata={"revision": {"latency_ms": 40}}),
    ]

    assert brier_score(rows) == pytest.approx(0.04)
    assert expected_calibration_error(rows) == pytest.approx(0.2)
    assert area_under_risk_coverage_curve(rows) == pytest.approx(0.125)
    assert accuracy_at_fixed_coverage(rows, target_coverage=0.5) == 1.0
    assert end_to_end_latency(rows) == 200
    assert revision_latency(rows) == 30
    assert tokens(rows) == 22.5


def test_supported_text_metrics_use_strict_text_label_not_structured_selection():
    from river.evaluation.metrics import supported_and_text_correct, supported_but_text_wrong

    rows = [
        {"citation_supported": True, "answer_correct": False, "answer_text_correct": True},
        {"citation_supported": True, "answer_correct": True, "answer_text_correct": False},
        {"citation_supported": False, "answer_correct": True, "answer_text_correct": False},
        {"citation_supported": True, "answer_correct": True, "answer_text_correct": None},
    ]
    assert supported_and_text_correct(rows) == pytest.approx(1 / 3)
    assert supported_but_text_wrong(rows) == pytest.approx(1 / 3)


def test_usr_and_cost_preconditions_mark_no_generation_runs_not_applicable():
    """A --no-generation run (e.g. MiniCheck without revision) names no selected
    hypothesis and records no generation cost. Table code must mark those cells
    "not applicable", not fail the whole table over a NaN mean.

    Without a sufficient Precondition on unsupported_supersession_rate and
    estimated_cost, such a run yields a NaN that aborts table generation
    instead of an empty cell.
    """
    from analysis.generate_tables import (
        METRIC_PRECONDITIONS,
        _NEEDS_COST_METADATA,
        _NEEDS_SELECTED_HYPOTHESIS_ON_LATE_ASSERTION,
    )

    no_generation_rows = [
        {
            "intervention": "unsupported_late_assertion",
            "intervention_hypothesis_id": "h1",
            "predicted_hypothesis_id": None,
            "answer_named_hypothesis_id": None,
            "metadata": {},
        }
    ]
    assert METRIC_PRECONDITIONS["unsupported_supersession_rate"] is _NEEDS_SELECTED_HYPOTHESIS_ON_LATE_ASSERTION
    assert METRIC_PRECONDITIONS["repetition_resistance"] is _NEEDS_SELECTED_HYPOTHESIS_ON_LATE_ASSERTION
    assert METRIC_PRECONDITIONS["estimated_cost"] is _NEEDS_COST_METADATA
    assert not _NEEDS_SELECTED_HYPOTHESIS_ON_LATE_ASSERTION.holds(no_generation_rows)
    assert not _NEEDS_COST_METADATA.holds(no_generation_rows)

    generation_rows = [
        {
            "intervention": "unsupported_late_assertion",
            "intervention_hypothesis_id": "h1",
            "predicted_hypothesis_id": "h1",
            "answer_named_hypothesis_id": None,
            "metadata": {"estimated_cost_usd": 0.0001},
        }
    ]
    assert _NEEDS_SELECTED_HYPOTHESIS_ON_LATE_ASSERTION.holds(generation_rows)
    assert _NEEDS_COST_METADATA.holds(generation_rows)
