import os
import sys

import pytest

from river import build_baseline
from river.baselines import BaselineUnavailable
from river.baselines.native import load_native_manifests
from river.baselines.native_cli import (
    command_report,
    credential_report,
    diagnose_baseline,
    list_availability,
    load_manifest,
)
from river.schemas import PredictionRecord


NATIVE_ADAPTERS = [
    ("self_rag", "controlled"),
    ("crag", "controlled"),
    ("graphrag", "controlled"),
    ("graphrag_local", "controlled"),
    ("graphrag_global", "native"),
    ("memgpt", "controlled"),
    ("zep", "controlled"),
]


def _runner(tmp_path, output=None):
    output = output or {
        "answer": "The native runner answered from the supplied evidence.",
        "predicted_state": "ACTIVE",
        "confidence": 0.7,
        "citation_supported": True,
        "input_tokens": 11,
        "output_tokens": 7,
    }
    script = tmp_path / "native_runner.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "payload = json.loads(sys.stdin.read())",
                "assert 'comparison_mode' in payload",
                "assert 'evidence' in payload",
                f"print(json.dumps({output!r}))",
            ]
        ),
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def _evidence():
    return [
        {
            "source_id": "doc-1",
            "chunk_id": "chunk-1",
            "text": "The native runner contract receives exact evidence chunks.",
        }
    ]


def test_native_manifests_cover_required_baselines():
    manifests = load_native_manifests()

    assert {
        "self_rag",
        "crag",
        "graphrag_local",
        "graphrag_global",
        "memgpt",
        "zep",
        "minicheck_no_revision",
    } <= set(manifests)
    for manifest in manifests.values():
        assert manifest.official_repository_url.startswith("https://github.com/")
        assert manifest.commit_or_tag
        assert manifest.paper_reference
        assert manifest.license
        assert manifest.installation_command
        assert manifest.isolation
        assert "stdin" in manifest.expected_output_mapping
        assert "stdout" in manifest.expected_output_mapping


@pytest.mark.parametrize(("baseline_id", "mode"), NATIVE_ADAPTERS)
def test_native_adapter_contract_invokes_configured_runner(tmp_path, baseline_id, mode):
    baseline = build_baseline(
        baseline_id,
        {
            "runner_command": _runner(tmp_path),
            "comparison_mode": mode,
            "allow_missing_credentials": True,
            "timeout_seconds": 5,
            "cost_per_input_token": 0.01,
            "cost_per_output_token": 0.02,
        },
    )

    result = baseline.process_turn("session-1", "What is supported?", _evidence(), target_time="2026-01-01T00:00:00")

    assert result["predicted_state"] == "active"
    assert result["comparison_mode"] == mode
    assert result["native_execution"]["state_source"] == "native"
    assert result["native_execution"]["attempts"] == 1
    assert result["native_execution"]["cost_usd"] == pytest.approx(0.25)
    assert result["native_execution"]["manifest"]["comparison_mode"] == mode


def test_native_adapter_does_not_invent_state_when_runner_omits_it(tmp_path):
    baseline = build_baseline(
        "self_rag",
        {
            "runner_command": _runner(tmp_path, {"answer": "No explicit state."}),
            "allow_missing_credentials": True,
        },
    )

    result = baseline.process_turn("session", "query", _evidence())

    assert result["predicted_state"] is None
    assert result["native_execution"]["state_source"] == "unsupported"


def test_native_adapter_marks_invalid_state_as_evaluator_derived(tmp_path):
    baseline = build_baseline(
        "self_rag",
        {
            "runner_command": _runner(tmp_path, {"answer": "Invalid state.", "state": "finished"}),
            "allow_missing_credentials": True,
        },
    )

    result = baseline.process_turn("session", "query", _evidence())

    assert result["predicted_state"] == "unknown"
    assert result["native_execution"]["state_source"] == "evaluator_derived"


def test_native_adapter_retries_failed_runner(tmp_path):
    state = tmp_path / "attempts.txt"
    script = tmp_path / "flaky_runner.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import pathlib",
                "import sys",
                f"state = pathlib.Path({str(state)!r})",
                "attempt = int(state.read_text() or '0') if state.exists() else 0",
                "state.write_text(str(attempt + 1))",
                "json.loads(sys.stdin.read())",
                "if attempt == 0:",
                "    print('temporary failure', file=sys.stderr)",
                "    raise SystemExit(2)",
                "print(json.dumps({'answer': 'ok', 'state': 'active'}))",
            ]
        ),
        encoding="utf-8",
    )
    baseline = build_baseline(
        "self_rag",
        {
            "runner_command": [sys.executable, str(script)],
            "allow_missing_credentials": True,
            "max_retries": 1,
        },
    )

    result = baseline.process_turn("session", "query", _evidence())

    assert result["predicted_state"] == "active"
    assert result["native_execution"]["attempts"] == 2


def test_native_adapter_timeout_fails_actionably(tmp_path):
    script = tmp_path / "sleep_runner.py"
    script.write_text("import time\ntime.sleep(2)\n", encoding="utf-8")
    baseline = build_baseline(
        "self_rag",
        {
            "runner_command": [sys.executable, str(script)],
            "allow_missing_credentials": True,
            "timeout_seconds": 0.01,
        },
    )

    with pytest.raises(BaselineUnavailable, match="timeout after"):
        baseline.process_turn("session", "query", _evidence())


def test_missing_runner_fails_with_actionable_setup_message():
    baseline = build_baseline("self_rag", {"allow_missing_credentials": True})

    with pytest.raises(BaselineUnavailable, match="no runner_command configured") as exc:
        baseline.process_turn("session", "query", _evidence())

    message = str(exc.value)
    assert "https://github.com/AkariAsai/self-rag" in message
    assert "Install command:" in message


def test_comparison_mode_is_explicit_and_enforced():
    with pytest.raises(BaselineUnavailable, match="comparison_mode='controlled'"):
        build_baseline("graphrag_global", {"comparison_mode": "controlled"})


def test_native_result_fits_prediction_record(tmp_path):
    baseline = build_baseline(
        "self_rag",
        {
            "runner_command": _runner(tmp_path),
            "allow_missing_credentials": True,
        },
    )
    result = baseline.process_turn("session", "query", _evidence())

    record = PredictionRecord(
        run_id="run",
        example_id="ex",
        base_example_id="base",
        dataset="river_bench",
        source_dataset="unit",
        split="test",
        intervention="order_reversal",
        comparison_mode=result["comparison_mode"],
        baseline="self_rag",
        baseline_version=baseline.version,
        generator="mock",
        seed=1,
        predicted_state=result["predicted_state"],
        citation_supported=result["citation_supported"],
        confidence=result["confidence"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        configuration_hash="sha256:config",
        code_commit="commit",
        dataset_hash="sha256:data",
        prompt_hash="sha256:prompt",
        metadata={"native_execution": result["native_execution"]},
    )

    assert record.comparison_mode == "controlled"
    assert record.metadata["native_execution"]["manifest"]["baseline_id"] == "self_rag"


def test_readiness_reports_are_offline_and_actionable(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    manifest = load_manifest("zep")

    credentials = credential_report(manifest)
    command = command_report(manifest)
    report = diagnose_baseline("zep")

    assert "NEO4J_URI" in credentials["missing"]
    assert command["ok"] is False
    assert report["ready"] is False
    assert "never opens a paid LLM call" in report["readiness_notes"]


def test_list_availability_includes_minicheck():
    rows = {row["baseline_id"]: row for row in list_availability()}

    assert rows["minicheck_no_revision"]["registered"] is True


@pytest.mark.native
def test_optional_native_smoke_when_configured():
    if os.getenv("RIVER_RUN_NATIVE_SMOKE") != "1":
        pytest.skip("set RIVER_RUN_NATIVE_SMOKE=1 and configure a native runner to execute this smoke test")
    baseline_id = os.environ["RIVER_NATIVE_BASELINE"]
    runner_command = os.environ["RIVER_NATIVE_RUNNER_COMMAND"]
    comparison_mode = os.getenv("RIVER_NATIVE_COMPARISON_MODE", "controlled")
    baseline = build_baseline(
        baseline_id,
        {
            "runner_command": runner_command,
            "comparison_mode": comparison_mode,
            "allow_missing_credentials": True,
            "timeout_seconds": float(os.getenv("RIVER_NATIVE_SMOKE_TIMEOUT_SECONDS", "30")),
        },
    )

    result = baseline.process_turn("smoke", "Return the supported statement.", _evidence())

    assert "native_execution" in result
    assert result["comparison_mode"] == comparison_mode
