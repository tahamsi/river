import json

import pytest

from analysis.generate_tables import (
    QUALITATIVE_ONLY_INTERVENTIONS,
    MissingDataError,
    TABLE_SPECS,
    generate_artifacts,
    latex_escape,
)


def _claim(label="supported", gold="supported"):
    return {
        "claim_id": f"claim-{label}-{gold}",
        "claim_text": "The claim is grounded.",
        "label": label,
        "supported_score": 0.9 if label == "supported" else 0.1,
        "citations": [{"source_id": "source", "chunk_id": "chunk"}],
        "metadata": {"gold_label": gold},
    }


def _record(
    *,
    baseline,
    example_id,
    pair_id,
    intervention,
    gold_state="active",
    predicted_state="active",
    gold_hypothesis_id="h_good",
    predicted_hypothesis_id="h_good",
    intervention_hypothesis_id=None,
    generator="model-a",
    threshold=0.4,
    run_id="run-model-a-04",
    source_dataset="fever",
    confidence=0.8,
    answer_correct=True,
    claim=None,
):
    metadata = {
        "threshold": threshold,
        "support_threshold": threshold,
        "revision": {"latency_ms": 12.0, "model_calls": 1},
        "model_calls": 2,
        "nli_calls": 3,
        "peak_memory_mb": 256.0,
        "estimated_cost_usd": 0.02,
        "human_audit": {
            "study_id": "audit-v1",
            "audit_time_seconds": 30.0,
            "issue_found": not answer_correct,
            "auditor_agreement": 0.8,
        },
        "annotation_quality": {
            "source": "annotation-v1",
            "label_agreement": 0.9,
            "fleiss_kappa": 0.75,
            "adjudicated": False,
        },
    }
    if baseline == "river_d":
        metadata.update({"condition": "river_d_full", "mechanism": "deterministic_revision", "ablation": "full"})
    elif baseline == "river_l":
        metadata.update({"condition": "river_l_full", "mechanism": "learned_revision", "ablation": "learned"})

    return {
        "run_id": run_id,
        "example_id": example_id,
        "base_example_id": example_id.split(":")[0],
        "pair_id": pair_id,
        "dataset": "river_bench_external_eval",
        "source_dataset": source_dataset,
        "split": "test",
        "intervention": intervention,
        "comparison_mode": "controlled",
        "baseline": baseline,
        "baseline_version": "1",
        "generator": generator,
        "seed": 1,
        "gold_state": gold_state,
        "gold_hypothesis_id": gold_hypothesis_id,
        "predicted_state": predicted_state,
        "predicted_hypothesis_id": predicted_hypothesis_id,
        "intervention_hypothesis_id": intervention_hypothesis_id,
        "answer_correct": answer_correct,
        # Mirrors the scorer: rows with no single correct answer are not scored,
        # so a conflict or abstain row carries None rather than a failure.
        "answer_text_correct": answer_correct if gold_hypothesis_id is not None else None,
        "citation_supported": claim is None or claim["label"] == "supported",
        "confidence": confidence,
        "state_probabilities": {
            "active": 0.8 if predicted_state == "active" else 0.1,
            "disputed": 0.8 if predicted_state == "disputed" else 0.1,
            "evolution": 0.8 if predicted_state == "evolution" else 0.1,
            "abstain": 0.8 if predicted_state == "abstain" else 0.1,
            "unknown": 0.05,
        },
        "claim_verification_results": [claim or _claim()],
        "latency_ms": 100.0,
        "input_tokens": 10,
        "output_tokens": 5,
        "configuration_hash": f"sha256:{baseline}-{threshold}",
        "code_commit": "commit",
        "dataset_hash": "sha256:data",
        "prompt_hash": "sha256:prompt",
        "metadata": metadata,
    }


def _baseline_records(baseline, *, generator="model-a", threshold=0.4):
    run_id = f"run-{baseline}-{generator}-{threshold}"
    suffix = f"{generator}:{threshold}"
    good = baseline != "naive_rag"
    rows = [
        _record(baseline=baseline, example_id=f"order:a:{suffix}", pair_id=f"p-order:{suffix}", intervention="order_reversal", generator=generator, threshold=threshold, run_id=run_id, predicted_hypothesis_id="h_good", answer_correct=True),
        _record(baseline=baseline, example_id=f"order:b:{suffix}", pair_id=f"p-order:{suffix}", intervention="order_reversal", generator=generator, threshold=threshold, run_id=run_id, predicted_hypothesis_id="h_good" if good else "h_bad", answer_correct=good),
        _record(baseline=baseline, example_id=f"update:a:{suffix}", pair_id=f"p-update:{suffix}", intervention="supported_genuine_update", generator=generator, threshold=threshold, run_id=run_id, gold_hypothesis_id="h_old", predicted_hypothesis_id="h_old", intervention_hypothesis_id="h_new", answer_correct=True),
        _record(baseline=baseline, example_id=f"update:b:{suffix}", pair_id=f"p-update:{suffix}", intervention="supported_genuine_update", generator=generator, threshold=threshold, gold_hypothesis_id="h_new", predicted_hypothesis_id="h_new" if good else "h_old", intervention_hypothesis_id="h_new", answer_correct=good, run_id=run_id),
        _record(baseline=baseline, example_id=f"unsupported:a:{suffix}", pair_id=f"p-unsupported:{suffix}", intervention="unsupported_late_assertion", generator=generator, threshold=threshold, run_id=run_id, predicted_hypothesis_id="h_good", intervention_hypothesis_id="h_bad", answer_correct=True),
        _record(baseline=baseline, example_id=f"unsupported:b:{suffix}", pair_id=f"p-unsupported:{suffix}", intervention="unsupported_late_assertion", generator=generator, threshold=threshold, run_id=run_id, predicted_hypothesis_id="h_good" if good else "h_bad", intervention_hypothesis_id="h_bad", answer_correct=good),
        _record(baseline=baseline, example_id=f"repeat:a:{suffix}", pair_id=f"p-repeat:{suffix}", intervention="repeated_weak_assertion", generator=generator, threshold=threshold, run_id=run_id, predicted_hypothesis_id="h_good", intervention_hypothesis_id="h_bad", answer_correct=True),
        _record(baseline=baseline, example_id=f"repeat:b:{suffix}", pair_id=f"p-repeat:{suffix}", intervention="repeated_weak_assertion", generator=generator, threshold=threshold, run_id=run_id, predicted_hypothesis_id="h_good" if good else "h_bad", intervention_hypothesis_id="h_bad", answer_correct=good),
        _record(baseline=baseline, example_id=f"retro:a:{suffix}", pair_id=f"p-retro:{suffix}", intervention="retroactive_correction", generator=generator, threshold=threshold, run_id=run_id, gold_hypothesis_id="h_fixed", predicted_hypothesis_id="h_fixed" if good else "h_old", intervention_hypothesis_id="h_fixed", answer_correct=good),
        _record(baseline=baseline, example_id=f"conflict:a:{suffix}", pair_id=f"p-conflict:{suffix}", intervention="concurrent_disagreement", generator=generator, threshold=threshold, run_id=run_id, gold_state="disputed", predicted_state="disputed" if good else "active", gold_hypothesis_id=None, predicted_hypothesis_id=None, intervention_hypothesis_id="h_conflict", answer_correct=good),
        _record(baseline=baseline, example_id=f"evolution:a:{suffix}", pair_id=f"p-evolution:{suffix}", intervention="temporal_evolution", generator=generator, threshold=threshold, run_id=run_id, gold_state="evolution", predicted_state="evolution" if good else "disputed", gold_hypothesis_id=None, predicted_hypothesis_id=None, intervention_hypothesis_id="h_new", answer_correct=good),
        _record(baseline=baseline, example_id=f"nei:a:{suffix}", pair_id=f"p-nei:{suffix}", intervention="insufficient_evidence_nei", generator=generator, threshold=threshold, run_id=run_id, gold_state="abstain", predicted_state="abstain" if good else "active", gold_hypothesis_id=None, predicted_hypothesis_id=None, intervention_hypothesis_id="h_claim", answer_correct=good),
        _record(baseline=baseline, example_id=f"removal:covered:{suffix}", pair_id=f"p-removal-covered:{suffix}", intervention="evidence_removal", generator=generator, threshold=threshold, run_id=run_id, gold_state="active", predicted_state="active", gold_hypothesis_id="h_good", predicted_hypothesis_id="h_good", intervention_hypothesis_id=None, answer_correct=True),
        _record(baseline=baseline, example_id=f"removal:a:{suffix}", pair_id=f"p-removal:{suffix}", intervention="evidence_removal", generator=generator, threshold=threshold, run_id=run_id, gold_state="abstain", predicted_state="abstain" if good else "active", gold_hypothesis_id=None, predicted_hypothesis_id=None, intervention_hypothesis_id="h_claim", answer_correct=good),
        _record(baseline=baseline, example_id=f"cite-support:{suffix}", pair_id=f"p-cite-support:{suffix}", intervention="citation_eval", generator=generator, threshold=threshold, run_id=run_id, claim=_claim("supported", "supported"), answer_correct=True),
        _record(baseline=baseline, example_id=f"cite-contradict:{suffix}", pair_id=f"p-cite-contradict:{suffix}", intervention="citation_eval", generator=generator, threshold=threshold, run_id=run_id, claim=_claim("contradicted", "contradicted"), answer_correct=False),
        _record(baseline=baseline, example_id=f"cite-neutral:{suffix}", pair_id=f"p-cite-neutral:{suffix}", intervention="citation_eval", generator=generator, threshold=threshold, run_id=run_id, claim=_claim("neutral", "neutral"), answer_correct=False),
        _record(baseline=baseline, example_id=f"cite-temporal:{suffix}", pair_id=f"p-cite-temporal:{suffix}", intervention="citation_eval", generator=generator, threshold=threshold, run_id=run_id, claim=_claim("temporal_mismatch", "temporal_mismatch"), answer_correct=False),
    ]
    for item in rows:
        if item["intervention"] == "supported_genuine_update" and item["gold_hypothesis_id"] == "h_new":
            item["metadata"]["stale_hypothesis_id"] = "h_old"
    return rows


def manuscript_records():
    rows = []
    for baseline in ("naive_rag", "river_d", "river_l"):
        rows.extend(_baseline_records(baseline, generator="model-a", threshold=0.4))
        rows.extend(_baseline_records(baseline, generator="model-b", threshold=0.6))
    return rows


def test_specs_cover_all_18_table_families():
    assert len(TABLE_SPECS) == 18
    for spec in TABLE_SPECS.values():
        assert spec.required_columns
        assert spec.filters
        assert spec.grouping
        assert spec.comparison_mode
        assert spec.directions is not None
        assert spec.confidence_intervals
        assert spec.statistical_test_source
        assert spec.caption
        assert spec.footnotes
        assert spec.minimum_required_data


def test_missing_data_failure_writes_report(tmp_path):
    with pytest.raises(MissingDataError):
        generate_artifacts([], tmp_path)

    report = json.loads((tmp_path / "missing_data_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "missing_data"
    assert "no prediction records found" in report["tables"]["all"][0]


def test_direction_aware_bolding_rounding_latex_and_json(tmp_path):
    records = manuscript_records()
    generate_artifacts(
        records,
        tmp_path,
        requested_tables=["overall_controlled_baseline_comparison"],
        requested_figures=[],
        bootstrap_resamples=20,
    )
    table_path = next((tmp_path / "tables").glob("table_01_*.md"))
    tex_path = next((tmp_path / "tables").glob("table_01_*.tex"))
    json_path = next((tmp_path / "tables").glob("table_01_*.json"))

    md = table_path.read_text(encoding="utf-8")
    tex = tex_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "**" in md
    assert "\\textbf{" in tex
    assert "0.000" in md or "1.000" in md
    assert "\\_" in latex_escape("river_d")
    json.dumps(payload, allow_nan=False)


def test_full_generation_outputs_numbers_provenance_tables_and_figures(tmp_path):
    generate_artifacts(manuscript_records(), tmp_path, bootstrap_resamples=20)

    numbers = json.loads((tmp_path / "manuscript" / "numbers.json").read_text(encoding="utf-8"))
    provenance = json.loads((tmp_path / "manuscript" / "provenance.json").read_text(encoding="utf-8"))

    assert len(numbers["tables"]) == 18
    assert {"risk_coverage", "threshold_sensitivity", "calibration", "per_intervention_effects"} <= set(numbers["figures"])
    assert "overall_controlled_baseline_comparison" in provenance["tables"]
    first_cells = provenance["tables"]["overall_controlled_baseline_comparison"]["cells"]
    assert first_cells
    assert all("prediction_files" in cell for cell in first_cells.values())
    assert len(list((tmp_path / "tables").glob("*.csv"))) == 18
    assert len(list((tmp_path / "tables").glob("*.json"))) == 18
    assert len(list((tmp_path / "tables").glob("*.md"))) == 18
    assert len(list((tmp_path / "tables").glob("*.tex"))) == 18
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4


def test_no_placeholder_or_engineering_values(tmp_path):
    records = manuscript_records()
    records[0]["source_dataset"] = "synthetic_debug"

    with pytest.raises(MissingDataError):
        generate_artifacts(records, tmp_path, requested_tables=["overall_controlled_baseline_comparison"], requested_figures=[])

    report = json.loads((tmp_path / "missing_data_report.json").read_text(encoding="utf-8"))
    assert records[0]["example_id"] in report["engineering_or_placeholder_rows"]


def test_reproducible_output_ordering(tmp_path):
    left = manuscript_records()
    right = list(reversed(manuscript_records()))

    generate_artifacts(left, tmp_path / "left", requested_tables=["overall_controlled_baseline_comparison"], requested_figures=[], bootstrap_resamples=10)
    generate_artifacts(right, tmp_path / "right", requested_tables=["overall_controlled_baseline_comparison"], requested_figures=[], bootstrap_resamples=10)

    left_json = next((tmp_path / "left" / "tables").glob("table_01_*.json")).read_text(encoding="utf-8")
    right_json = next((tmp_path / "right" / "tables").glob("table_01_*.json")).read_text(encoding="utf-8")
    assert left_json == right_json


def _make_stateless(records, baseline="naive_rag"):
    """Strip everything a system without a revision operator would never emit."""

    for row in records:
        if row["baseline"] != baseline:
            continue
        row["predicted_state"] = None
        row["predicted_hypothesis_id"] = None
        row["confidence"] = None
        row["state_probabilities"] = None
    return records


def test_undefined_metric_is_not_applicable_rather_than_missing(tmp_path):
    records = _make_stateless(manuscript_records())

    generate_artifacts(
        records,
        tmp_path,
        requested_tables=["nei_selective_abstention"],
        requested_figures=[],
        bootstrap_resamples=20,
    )

    md = next((tmp_path / "tables").glob("*nei_selective_abstention.md")).read_text(encoding="utf-8")
    payload = json.loads(next((tmp_path / "tables").glob("*nei_selective_abstention.json")).read_text(encoding="utf-8"))

    # The table is produced, rather than the whole family being suppressed
    # because one system has no revision state to score.
    assert "naive_rag" in md
    assert "is not applicable for" in md
    naive_row = next(row for row in payload["rows"] if row["baseline"] == "naive_rag")
    assert naive_row["f1"] is None
    assert naive_row["coverage"] is None
    river_row = next(row for row in payload["rows"] if row["baseline"] == "river_l")
    assert isinstance(river_row["f1"], float)
    reasons = payload["not_applicable"]["naive_rag"]
    assert "revision state" in reasons["f1"]


def test_absent_telemetry_still_suppresses_the_table(tmp_path):
    records = manuscript_records()
    for row in records:
        if row["baseline"] != "naive_rag":
            continue
        row["input_tokens"] = None
        row["output_tokens"] = None
        for key in ("model_calls", "nli_calls", "revision"):
            row["metadata"].pop(key, None)

    # Telemetry a baseline should have recorded and did not is a gap, not a
    # property of the system, so it must still stop the table being written.
    with pytest.raises(MissingDataError):
        generate_artifacts(
            records,
            tmp_path,
            requested_tables=["computational_cost"],
            requested_figures=[],
            bootstrap_resamples=20,
        )

    report = json.loads((tmp_path / "missing_data_report.json").read_text(encoding="utf-8"))
    assert any("tokens" in note for note in report["tables"]["computational_cost"])


def test_study_that_was_never_run_is_absent_not_missing(tmp_path):
    records = manuscript_records()
    for row in records:
        row["metadata"].pop("human_audit", None)
        row["metadata"].pop("annotation_quality", None)

    # A table reporting on a study nobody ran must not hold back the sixteen
    # tables whose data does exist, but it must still be named in the output.
    summary = generate_artifacts(records, tmp_path, bootstrap_resamples=20, requested_figures=[])

    absent = summary["numbers"]["absent_by_design"] if "numbers" in summary else None
    numbers = json.loads((tmp_path / "manuscript" / "numbers.json").read_text(encoding="utf-8"))
    assert set(numbers["absent_by_design"]) == {"human_auditing_study", "annotation_quality"}
    assert len(numbers["tables"]) == 16
    assert len(list((tmp_path / "tables").glob("*.md"))) == 16
    del absent


def test_ablation_variants_stay_out_of_the_headline_comparison(tmp_path):
    """An ablation is not a system under comparison.

    Pooling one into the baseline it ablates inflates that baseline's sample and
    drags its metrics toward the variant. When four RIVER-L ablations first ran,
    the headline row covered nine conditions at once and its recency flip rate
    moved from 0.000 to 0.002 purely because the no-temporal-features variant
    was averaged in.
    """

    records = manuscript_records()
    variant = [dict(row) for row in records if row["baseline"] == "river_l"]
    for row in variant:
        row["example_id"] = f"ablated:{row['example_id']}"
        row["pair_id"] = f"ablated:{row['pair_id']}"
        row["metadata"] = {**row["metadata"], "condition": "river_l_minus_temporal", "ablation": "minus_temporal"}
    records.extend(variant)
    pooled_without_the_guard = sum(
        1
        for row in records
        if row["baseline"] == "river_l" and row["intervention"] not in QUALITATIVE_ONLY_INTERVENTIONS
    )

    generate_artifacts(
        records,
        tmp_path,
        requested_tables=["overall_controlled_baseline_comparison", "main_river_ablation"],
        requested_figures=[],
        bootstrap_resamples=20,
    )

    headline = json.loads(
        next((tmp_path / "tables").glob("*overall_controlled_baseline_comparison.json")).read_text(encoding="utf-8")
    )
    river_l = next(row for row in headline["rows"] if row["baseline"] == "river_l")
    # Every ablated row is a copy of an unablated one, so admitting them would
    # exactly double the sample this row rests on.
    assert river_l["n"] == pooled_without_the_guard // 2

    # The variant still has to appear where it is the subject.
    ablation = json.loads(next((tmp_path / "tables").glob("*main_river_ablation.json")).read_text(encoding="utf-8"))
    assert any(row["condition"] == "river_l_minus_temporal" for row in ablation["rows"])
