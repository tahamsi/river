import json

import pytest

from river.evaluation.cli import evaluate_primary_endpoints, main as evaluation_main
from river.evaluation.metrics import EvaluationError


def prediction(example_id, baseline, correct=True, **overrides):
    payload = {
        "run_id": "run",
        "example_id": example_id,
        "base_example_id": example_id,
        "pair_id": "pair-" + example_id,
        "dataset": "river_bench",
        "source_dataset": "unit",
        "split": "test",
        "intervention": "supported_genuine_update",
        "comparison_mode": "controlled",
        "baseline": baseline,
        "baseline_version": "1",
        "generator": "mock",
        "seed": 1,
        "gold_state": "active",
        "gold_hypothesis_id": "h_new",
        "predicted_state": "active",
        "predicted_hypothesis_id": "h_new" if correct else "h_old",
        "intervention_hypothesis_id": "h_new",
        "configuration_hash": "sha256:config",
        "code_commit": "commit",
        "dataset_hash": "sha256:data",
        "prompt_hash": "sha256:prompt",
    }
    payload.update(overrides)
    return payload


def test_primary_endpoint_analysis_returns_machine_and_manuscript_data():
    records = [
        prediction("ex1", "river_l", True),
        prediction("ex2", "river_l", True),
        prediction("ex1", "river_d", True),
        prediction("ex2", "river_d", False),
    ]

    result = evaluate_primary_endpoints(
        records,
        candidate_baseline="river_l",
        reference_baseline="river_d",
        bootstrap_resamples=100,
    )

    assert result["schema_version"] == "river-evaluation-v1"
    assert len(result["metrics"]) == 2
    assert result["paired_statistics"]["candidate_baseline"] == "river_l"
    assert result["paired_statistics"]["revision_correctness"]["bootstrap_ci"]["estimate"] == 0.5


def test_primary_endpoint_cli_writes_json_and_markdown(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        prediction("ex1", "river_l", True),
        prediction("ex1", "river_d", False),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "metrics"

    code = evaluation_main([
        "primary-endpoints",
        str(path),
        "--output",
        str(output),
        "--candidate-baseline",
        "river_l",
        "--reference-baseline",
        "river_d",
        "--bootstrap-resamples",
        "50",
    ])

    assert code == 0
    payload = json.loads((output / "primary_endpoints.json").read_text(encoding="utf-8"))
    assert payload["paired_statistics"]["candidate_baseline"] == "river_l"
    assert "# RIVER Primary Endpoint Evaluation" in (output / "primary_endpoints.md").read_text(encoding="utf-8")


def test_primary_endpoint_refuses_mode_and_split_mixing():
    with pytest.raises(EvaluationError, match="controlled/native"):
        evaluate_primary_endpoints([
            prediction("ex1", "river_d", comparison_mode="controlled"),
            prediction("ex2", "river_d", comparison_mode="native"),
        ])

    with pytest.raises(EvaluationError, match="mixed calibration"):
        evaluate_primary_endpoints([
            prediction("ex1", "river_d", split="validation"),
            prediction("ex2", "river_d", split="test"),
        ])
