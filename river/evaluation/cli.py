"""Command-line evaluation for normalized RIVER prediction records."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import (
    PAIRED_INTERVENTIONS,
    SCALAR_METRICS,
    STRUCTURED_METRICS,
    EvaluationError,
    validate_no_calibration_test_mix,
    validate_required_pair_ids,
    validate_single_comparison_mode,
)
from .statistics import (
    align_by_example,
    benjamini_hochberg_correction,
    effect_sizes,
    genuine_update_non_inferiority,
    holm_correction,
    mcnemar_test,
    paired_bootstrap_ci,
    paired_permutation_test,
    revision_correct,
    seed_aggregation,
)


PRIMARY_ENDPOINT_METRICS = [
    "answer_text_accuracy",
    "end_to_end_answer_accuracy",
    "recency_flip_rate",
    "probability_recency_sensitivity",
    "unsupported_supersession_rate",
    "order_reversal_consistency",
    "repetition_resistance",
    "stable_and_correct",
    "stable_but_wrong",
    "flipped_to_correct",
    "flipped_to_wrong",
    "genuine_update_accuracy",
    "retroactive_correction_accuracy",
    "historical_state_accuracy",
    "current_state_accuracy",
    "stale_belief_rate",
    "conflict_precision",
    "conflict_recall",
    "conflict_macro_f1",
    "conflict_specificity",
    "false_conflict_rate",
    "abstention_precision",
    "abstention_recall",
    "abstention_f1",
    "selective_risk",
    "coverage",
    "area_under_risk_coverage_curve",
    "accuracy_at_fixed_coverage",
    "citation_compliance_rate",
    "citation_support_precision",
    "citation_support_recall",
    "citation_support_f1",
    "contradiction_f1",
    "neutral_insufficient_f1",
    "temporal_mismatch_f1",
    "supported_and_correct",
    "compliant_but_unsupported",
    "supported_but_wrong",
    "supported_and_text_correct",
    "supported_but_text_wrong",
    "brier_score",
    "expected_calibration_error",
    "end_to_end_latency",
    "revision_latency",
    "tokens",
    "model_calls",
    "nli_calls",
    "peak_memory",
    "estimated_cost",
]


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.is_dir():
            records.extend(load_records(sorted(path.rglob("*.jsonl"))))
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("_type") != "run_config":
                records.append(item)
    return records


def filter_records(
    records: list[dict[str, Any]],
    *,
    splits: list[str] | None = None,
    comparison_mode: str | None = None,
    baselines: list[str] | None = None,
) -> list[dict[str, Any]]:
    filtered = records
    if splits:
        wanted_splits = {split.lower() for split in splits}
        filtered = [row for row in filtered if str(row.get("split", "")).lower() in wanted_splits]
    if comparison_mode:
        filtered = [row for row in filtered if row.get("comparison_mode", "controlled") == comparison_mode]
    if baselines:
        wanted_baselines = set(baselines)
        filtered = [row for row in filtered if row.get("baseline") in wanted_baselines]
    return filtered


def _group_by_baseline_mode(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row.get("baseline", "unknown"), row.get("comparison_mode", "controlled"))].append(row)
    return grouped


def _compute_scalar_metrics(records: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in metric_names:
        values[name] = SCALAR_METRICS[name](records)
    return values


def _compute_structured_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {name: function(records) for name, function in STRUCTURED_METRICS.items()}


def evaluate_primary_endpoints(
    records: list[dict[str, Any]],
    *,
    reference_baseline: str | None = None,
    candidate_baseline: str | None = None,
    non_inferiority_margin: float = 0.02,
    final_test: bool = True,
    p_correction: str = "holm",
    bootstrap_resamples: int = 2000,
    seed: int = 13,
) -> dict[str, Any]:
    if not records:
        raise EvaluationError("no prediction records available for evaluation")
    validate_single_comparison_mode(records)
    if final_test:
        validate_no_calibration_test_mix(records)
    present_paired = {row.get("intervention") for row in records} & PAIRED_INTERVENTIONS
    if present_paired:
        validate_required_pair_ids(records, present_paired)

    grouped = _group_by_baseline_mode(records)
    metric_rows: list[dict[str, Any]] = []
    structured: dict[str, Any] = {}
    for (baseline, mode), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        metric_row: dict[str, Any] = {
            "baseline": baseline,
            "comparison_mode": mode,
            "n": len(items),
        }
        metric_row.update(_compute_scalar_metrics(items, PRIMARY_ENDPOINT_METRICS))
        metric_rows.append(metric_row)
        structured[f"{baseline}:{mode}"] = _compute_structured_metrics(items)

    paired_statistics: dict[str, Any] = {}
    if reference_baseline and candidate_baseline:
        paired_statistics = _paired_baseline_statistics(
            records,
            reference_baseline=reference_baseline,
            candidate_baseline=candidate_baseline,
            non_inferiority_margin=non_inferiority_margin,
            p_correction=p_correction,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )

    return {
        "schema_version": "river-evaluation-v1",
        "n_records": len(records),
        "comparison_mode": validate_single_comparison_mode(records),
        "splits": sorted({row.get("split", "unknown") for row in records}),
        "metrics": metric_rows,
        "structured_metrics": structured,
        "seed_aggregation": seed_aggregation(records, [
            "recency_flip_rate",
            "genuine_update_accuracy",
            "unsupported_supersession_rate",
            "conflict_macro_f1",
            "abstention_f1",
        ]),
        "paired_statistics": paired_statistics,
        "warnings": [
            "Metrics with missing required labels are reported as null in JSON and NA in Markdown.",
            "Citation precision/recall/F1 require explicit claim gold labels; verifier outputs alone are not ground truth.",
        ],
    }


def _paired_baseline_statistics(
    records: list[dict[str, Any]],
    *,
    reference_baseline: str,
    candidate_baseline: str,
    non_inferiority_margin: float,
    p_correction: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    candidate_rows, reference_rows = align_by_example(records, candidate_baseline, reference_baseline)
    candidate_correct = [revision_correct(row) for row in candidate_rows]
    reference_correct = [revision_correct(row) for row in reference_rows]

    stats: dict[str, Any] = {
        "candidate_baseline": candidate_baseline,
        "reference_baseline": reference_baseline,
        "revision_correctness": {
            "bootstrap_ci": paired_bootstrap_ci(
                candidate_correct,
                reference_correct,
                n_resamples=bootstrap_resamples,
                seed=seed,
            ),
            "mcnemar": mcnemar_test(candidate_correct, reference_correct),
            "paired_permutation": paired_permutation_test(candidate_correct, reference_correct, seed=seed),
            "effect_sizes": effect_sizes(candidate_correct, reference_correct),
        },
        "genuine_update_non_inferiority": genuine_update_non_inferiority(
            records,
            candidate_baseline=candidate_baseline,
            reference_baseline=reference_baseline,
            margin=non_inferiority_margin,
            seed=seed,
        ),
    }
    p_values = {
        "revision_correctness_mcnemar": stats["revision_correctness"]["mcnemar"]["p_value"],
        "revision_correctness_permutation": stats["revision_correctness"]["paired_permutation"]["p_value"],
        "genuine_update_non_inferiority": stats["genuine_update_non_inferiority"]["p_value"],
    }
    if p_correction == "holm":
        adjusted = holm_correction(p_values)
    elif p_correction in {"bh", "benjamini-hochberg", "benjamini_hochberg"}:
        adjusted = benjamini_hochberg_correction(p_values)
    else:
        raise ValueError("p_correction must be holm or bh")
    stats["p_value_correction"] = {"method": p_correction, "adjusted_p_values": adjusted}
    return stats


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "NA" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def manuscript_markdown(result: dict[str, Any]) -> str:
    metric_rows = result["metrics"]
    if not metric_rows:
        return "# RIVER Evaluation\n\nNo metric rows.\n"
    fields = ["baseline", "comparison_mode", "n", *PRIMARY_ENDPOINT_METRICS]
    lines = [
        "# RIVER Primary Endpoint Evaluation",
        "",
        f"Records: {result['n_records']}",
        f"Splits: {', '.join(str(item) for item in result['splits'])}",
        f"Comparison mode: {result.get('comparison_mode') or 'unspecified'}",
        "",
        "## Primary Metrics",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in metric_rows:
        lines.append("| " + " | ".join(_format_value(row.get(field)) for field in fields) + " |")

    if result.get("paired_statistics"):
        lines.extend(["", "## Paired Statistics", ""])
        stats = result["paired_statistics"]
        lines.append(f"Candidate: `{stats['candidate_baseline']}`")
        lines.append(f"Reference: `{stats['reference_baseline']}`")
        revision = stats["revision_correctness"]
        ci = revision["bootstrap_ci"]
        lines.append(
            "Revision correctness difference: "
            f"{_format_value(ci['estimate'])} "
            f"[{_format_value(ci['ci_low'])}, {_format_value(ci['ci_high'])}]"
        )
        lines.append(f"McNemar p-value: {_format_value(revision['mcnemar']['p_value'])}")
        lines.append(f"Permutation p-value: {_format_value(revision['paired_permutation']['p_value'])}")
        ni = stats["genuine_update_non_inferiority"]
        lines.append(
            "GUA non-inferiority: "
            f"observed diff {_format_value(ni['observed_difference'])}, "
            f"margin {_format_value(ni['margin'])}, p={_format_value(ni['p_value'])}, "
            f"non-inferior={ni['non_inferior']}"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {warning}" for warning in result.get("warnings", []))
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate normalized RIVER prediction records.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    primary = subparsers.add_parser("primary-endpoints", help="Compute primary endpoint metrics and statistics.")
    primary.add_argument("inputs", nargs="+", type=Path)
    primary.add_argument("--output", type=Path, default=Path("artifacts/metrics/river_primary_endpoints"))
    primary.add_argument("--split", action="append", help="Filter to a split. Repeatable.")
    primary.add_argument("--comparison-mode", choices=["controlled", "native"])
    primary.add_argument("--baseline", action="append", help="Filter to a baseline. Repeatable.")
    primary.add_argument("--reference-baseline")
    primary.add_argument("--candidate-baseline")
    primary.add_argument("--non-inferiority-margin", type=float, default=0.02)
    primary.add_argument("--p-correction", choices=["holm", "bh", "benjamini-hochberg"], default="holm")
    primary.add_argument("--bootstrap-resamples", type=int, default=2000)
    primary.add_argument("--seed", type=int, default=13)
    primary.add_argument("--exploratory", action="store_true", help="Allow non-final mixed split analysis.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "primary-endpoints":
        records = filter_records(
            load_records(args.inputs),
            splits=args.split,
            comparison_mode=args.comparison_mode,
            baselines=args.baseline,
        )
        result = evaluate_primary_endpoints(
            records,
            reference_baseline=args.reference_baseline,
            candidate_baseline=args.candidate_baseline,
            non_inferiority_margin=args.non_inferiority_margin,
            final_test=not args.exploratory,
            p_correction=args.p_correction,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "primary_endpoints.json").write_text(
            json.dumps(_json_safe(result), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (args.output / "primary_endpoints.md").write_text(manuscript_markdown(result), encoding="utf-8")
        print(args.output / "primary_endpoints.json")
        print(args.output / "primary_endpoints.md")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
