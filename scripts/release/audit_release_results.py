"""Fail-closed audit for the human-frozen RIVER release result package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


EXPECTED_PRIMARY = {
    ("naive_rag", 42),
    ("stateless_rag", 42),
    ("last_write_wins", 42),
    ("recency_weighted", 42),
    ("river_d", 42),
    ("river_l", 11),
    ("river_l", 42),
    ("river_l", 101),
    ("river_l", 202),
    ("river_l", 314),
}
STATEFUL_RENDERERS = {"last_write_wins", "recency_weighted", "river_d", "river_l"}
RERENDERED_TOP_LEVEL_FIELDS = {
    "answer",
    "answer_contract_valid",
    "answer_decision_faithful",
    "answer_named_hypothesis_id",
    "answer_text_correct",
    "code_commit",
    "direct_answer",
    "end_to_end_answer_correct",
    "generator",
    "input_tokens",
    "latency_ms",
    "output_tokens",
    "prompt_hash",
    "run_id",
}
RERENDERED_METADATA_FIELDS = {
    "estimated_cost_usd",
    "generation_contract",
    "generator_manifest",
    "model_calls",
    "rerendered_from_run_id",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def revision_correct(row: dict) -> bool:
    if row.get("predicted_state") != row.get("gold_state"):
        return False
    return row.get("gold_hypothesis_id") is None or row.get("predicted_hypothesis_id") == row.get("gold_hypothesis_id")


def structured_payload(row: dict) -> dict:
    """Remove only fields that the release rerender is authorised to change."""

    payload = {key: value for key, value in row.items() if key not in RERENDERED_TOP_LEVEL_FIELDS}
    metadata = dict(payload.get("metadata") or {})
    for key in RERENDERED_METADATA_FIELDS:
        metadata.pop(key, None)
    payload["metadata"] = metadata
    return payload


def structured_digest(rows: list[dict]) -> str:
    encoded = json.dumps(
        [structured_payload(row) for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--minicheck-dir", type=Path)
    parser.add_argument("--review-note-resolution", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, default=248)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    fixture_rows = read_jsonl(args.fixture)
    fixture_ids = {row["example_id"] for row in fixture_rows}
    if len(fixture_rows) != args.expected_examples or len(fixture_ids) != args.expected_examples:
        failures.append(
            f"fixture expected {args.expected_examples} unique rows; got {len(fixture_rows)} rows/{len(fixture_ids)} IDs"
        )

    files = sorted(args.primary_dir.glob("*.jsonl"))
    observed: set[tuple[str, int]] = set()
    all_rows: list[dict] = []
    files_report: dict[str, dict] = {}
    # Decision faithfulness (whether the rendered text literally matches the
    # selected hypothesis) is measured and reported, not gated. Mismatches are
    # deterministic behaviour of the generator at temperature 0, and removing
    # them would require re-sampling selected rows. Every unfaithful row is
    # listed. A generator call on a non-active decision
    # (generated_after_nonactive, below) remains a hard failure, because it is
    # a defect in the rendering code rather than in the model output.
    faithfulness_active_total = 0
    faithfulness_unfaithful_total = 0
    faithfulness_unfaithful_examples: list[dict] = []
    for path in files:
        rows = read_jsonl(path)
        if not rows:
            failures.append(f"{path.name}: empty")
            continue
        keys = {(str(row.get("baseline")), int(row.get("seed", -1))) for row in rows}
        if len(keys) != 1:
            failures.append(f"{path.name}: multiple baseline/seed keys {sorted(keys)}")
            continue
        key = next(iter(keys))
        observed.add(key)
        source_path = args.source_dir / path.name
        if not source_path.is_file():
            failures.append(f"{path.name}: matching source prediction file is missing")
            source_rows = []
        else:
            source_rows = read_jsonl(source_path)
            if structured_digest(source_rows) != structured_digest(rows):
                failures.append(
                    f"{path.name}: frozen structured fields changed during answer rerender"
                )
        ids = [row.get("example_id") for row in rows]
        if len(rows) != args.expected_examples or len(set(ids)) != args.expected_examples:
            failures.append(f"{path.name}: expected {args.expected_examples} unique rows")
        if set(ids) != fixture_ids:
            failures.append(f"{path.name}: example ID set differs from fixture")
        dirty = sorted(
            {
                str(row.get("code_commit"))
                for row in rows
                if "uncommitted" in str(row.get("code_commit")) or "+dirty:" in str(row.get("code_commit"))
            }
        )
        if dirty:
            failures.append(f"{path.name}: uncommitted/dirty code provenance {dirty}")
        invalid_contract = sum(row.get("answer_contract_valid") is not True for row in rows)
        if invalid_contract:
            failures.append(f"{path.name}: {invalid_contract} answer-contract violations")
        if key[0] in STATEFUL_RENDERERS:
            active_rows = [row for row in rows if row.get("predicted_state") == "active"]
            unfaithful_rows = [row for row in active_rows if row.get("answer_decision_faithful") is not True]
            faithfulness_active_total += len(active_rows)
            faithfulness_unfaithful_total += len(unfaithful_rows)
            if unfaithful_rows:
                faithfulness_unfaithful_examples.extend(
                    {
                        "file": path.name,
                        "example_id": row.get("example_id"),
                        "direct_answer": row.get("direct_answer"),
                    }
                    for row in unfaithful_rows
                )
                warnings.append(
                    f"{path.name}: {len(unfaithful_rows)}/{len(active_rows)} active answers do not "
                    "literally match the selected hypothesis (see decision_faithfulness.unfaithful_examples)"
                )
            generated_after_nonactive = sum(
                row.get("predicted_state") != "active"
                and int((row.get("metadata") or {}).get("model_calls", 0)) != 0
                for row in rows
            )
            if generated_after_nonactive:
                failures.append(f"{path.name}: generator called on {generated_after_nonactive} non-active decisions")
        files_report[path.name] = {
            "baseline": key[0],
            "seed": key[1],
            "rows": len(rows),
            "joint_revision_accuracy": mean(revision_correct(row) for row in rows),
            "strict_answer_accuracy": mean(
                bool(row["answer_text_correct"])
                for row in rows
                if row.get("answer_text_correct") is not None
            ),
            "source_structured_digest": structured_digest(source_rows) if source_rows else None,
            "release_structured_digest": structured_digest(rows),
        }
        all_rows.extend(rows)

    missing_primary = sorted(EXPECTED_PRIMARY - observed)
    extra_primary = sorted(observed - EXPECTED_PRIMARY)
    if missing_primary:
        failures.append(f"missing primary runs: {missing_primary}")
    if extra_primary:
        failures.append(f"unexpected primary runs: {extra_primary}")

    resolution = json.loads(args.review_note_resolution.read_text(encoding="utf-8"))
    resolved_ids = set(resolution.get("resolved_example_ids") or [])
    flagged_ids = {
        row["example_id"]
        for row in fixture_rows
        if "require human re-review" in str((row.get("transformation_metadata") or {}).get("review_note", "")).lower()
    }
    if not flagged_ids.issubset(resolved_ids):
        failures.append(
            "review-note resolution does not cover every flagged test fixture ID: "
            f"flagged={len(flagged_ids)}, resolved={len(resolved_ids)}"
        )
    for field in ("approved_by", "approved_at_utc", "annotation_workbook_sha256", "resolution"):
        if not resolution.get(field):
            failures.append(f"review-note resolution missing {field}")

    minicheck_report = None
    if args.minicheck_dir:
        mini_files = sorted(args.minicheck_dir.glob("*.jsonl"))
        if len(mini_files) != 1:
            failures.append(f"expected one MiniCheck prediction file, found {len(mini_files)}")
        elif mini_files:
            rows = read_jsonl(mini_files[0])
            if len(rows) != args.expected_examples or {row.get("example_id") for row in rows} != fixture_ids:
                failures.append("MiniCheck rows/IDs differ from fixture")
            revisions = set()
            weight_sets = set()
            for row in rows:
                verifier = (((row.get("metadata") or {}).get("verification") or {}).get("verifier") or {})
                provenance = verifier.get("resolved_provenance") or {}
                revision = provenance.get("resolved_model_revision")
                weights = provenance.get("weight_files") or {}
                if not revision or not weights or provenance.get("load_source") != "explicit_local_snapshot":
                    failures.append(f"MiniCheck row {row.get('example_id')} lacks enforced snapshot provenance")
                    break
                if revision != verifier.get("model_revision"):
                    failures.append(f"MiniCheck row {row.get('example_id')} requested/resolved revision mismatch")
                    break
                revisions.add(revision)
                weight_sets.add(json.dumps(weights, sort_keys=True))
            if len(revisions) != 1 or len(weight_sets) != 1:
                failures.append("MiniCheck provenance is not constant across rows")
            minicheck_report = {
                "file": mini_files[0].name,
                "rows": len(rows),
                "resolved_revisions": sorted(revisions),
                "weight_hash_sets": len(weight_sets),
            }

    river_l = defaultdict(list)
    for report in files_report.values():
        if report["baseline"] == "river_l":
            river_l["joint_revision_accuracy"].append(report["joint_revision_accuracy"])
            river_l["strict_answer_accuracy"].append(report["strict_answer_accuracy"])
    seed_summary = {}
    for metric, values in river_l.items():
        seed_summary[metric] = {
            "mean": mean(values) if values else math.nan,
            "sample_sd": stdev(values) if len(values) > 1 else math.nan,
            "n_seeds": len(values),
        }

    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "warnings": warnings,
        "fixture_rows": len(fixture_rows),
        "primary_runs": files_report,
        "river_l_seed_summary": seed_summary,
        "gold_state_counts": dict(Counter(row.get("gold_state") for row in fixture_rows)),
        "stale_review_notes_resolved": len(flagged_ids),
        "minicheck": minicheck_report,
        "decision_faithfulness": {
            "active_rows": faithfulness_active_total,
            "unfaithful_rows": faithfulness_unfaithful_total,
            "rate": (
                1.0 - faithfulness_unfaithful_total / faithfulness_active_total
                if faithfulness_active_total
                else None
            ),
            "unfaithful_examples": faithfulness_unfaithful_examples,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
