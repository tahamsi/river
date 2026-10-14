import json
from copy import deepcopy
from pathlib import Path

from scripts.release.audit_release_results import main as audit_main
from scripts.release.audit_release_results import structured_digest


def test_structured_digest_ignores_only_authorised_rerender_fields():
    source = {
        "example_id": "example-1",
        "predicted_state": "active",
        "predicted_hypothesis_id": "h1",
        "answer": "old",
        "run_id": "old-run",
        "code_commit": "old-commit",
        "metadata": {
            "revision": {"supporting_evidence_ids": ["e1"]},
            "model_calls": 1,
            "generator_manifest": {"model": "old"},
        },
    }
    rerendered = deepcopy(source)
    rerendered.update(
        answer="FINAL_ANSWER: new",
        run_id="new-run",
        code_commit="new-commit",
        direct_answer="new",
        answer_contract_valid=True,
    )
    rerendered["metadata"].update(
        model_calls=1,
        generator_manifest={"model": "new"},
        generation_contract="river-decision-render-v2",
        rerendered_from_run_id="old-run",
    )
    assert structured_digest([source]) == structured_digest([rerendered])


def test_structured_digest_detects_changed_decision():
    source = {
        "example_id": "example-1",
        "predicted_state": "active",
        "predicted_hypothesis_id": "h1",
        "metadata": {"revision": {"supporting_evidence_ids": ["e1"]}},
    }
    changed = deepcopy(source)
    changed["predicted_hypothesis_id"] = "h2"
    assert structured_digest([source]) != structured_digest([changed])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_unfaithful_active_answer_is_reported_not_gated(tmp_path, monkeypatch):
    """A deterministic, temperature-0 mismatch between the rendered text and the
    selected hypothesis is real model behaviour, not a code defect: it must be
    named in decision_faithfulness and must not fail the release on its own.
    A generator call on a non-active decision is a contract violation and must
    still fail the release.
    """
    fixture_row = {
        "example_id": "ex-1",
        "gold_state": "active",
        "gold_hypothesis_id": "h1",
        "transformation_metadata": {},
    }
    fixture_path = tmp_path / "fixture.jsonl"
    _write_jsonl(fixture_path, [fixture_row])

    source_dir = tmp_path / "source"
    primary_dir = tmp_path / "primary"
    source_dir.mkdir()
    primary_dir.mkdir()

    row = {
        "example_id": "ex-1",
        "baseline": "last_write_wins",
        "seed": 42,
        "predicted_state": "active",
        "predicted_hypothesis_id": "h1",
        "answer_decision_faithful": False,
        "answer_contract_valid": True,
        "answer_text_correct": False,
        "code_commit": "abc123",
        "metadata": {"model_calls": 1},
    }
    _write_jsonl(source_dir / "last_write_wins_test_seed42.jsonl", [row])
    _write_jsonl(primary_dir / "last_write_wins_test_seed42.jsonl", [row])

    resolution = tmp_path / "review_note_resolution.json"
    resolution.write_text(
        json.dumps(
            {
                "resolved_example_ids": [],
                "approved_by": "Reviewer",
                "approved_at_utc": "2026-01-01T00:00:00Z",
                "annotation_workbook_sha256": "deadbeef",
                "resolution": "ok",
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_release_results.py",
            "--fixture", str(fixture_path),
            "--source-dir", str(source_dir),
            "--primary-dir", str(primary_dir),
            "--review-note-resolution", str(resolution),
            "--expected-examples", "1",
            "--output", str(output),
        ],
    )
    exit_code = audit_main()
    report = json.loads(output.read_text())
    assert not any("do not match the selected hypothesis" in f for f in report["failures"])
    assert report["decision_faithfulness"]["active_rows"] == 1
    assert report["decision_faithfulness"]["unfaithful_rows"] == 1
    assert report["decision_faithfulness"]["unfaithful_examples"][0]["example_id"] == "ex-1"
    assert any("do not literally match" in w for w in report["warnings"])
    # Still gated: EXPECTED_PRIMARY mismatch is the only reason this minimal
    # fixture fails, not the faithfulness check.
    assert exit_code != 0
    assert not any("do not match the selected hypothesis" in f for f in report["failures"])
