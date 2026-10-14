"""Generate traceable manuscript tables and figures from PredictionRecord JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from river.evaluation.metrics import (
    QUALITATIVE_ONLY_INTERVENTIONS,
    SCALAR_METRICS,
    EvaluationError,
    answer_or_revision_correct,
    conflict_evolution_confusion_matrix,
    reliability_bins,
    revision_correct,
    risk_coverage_curve,
)
from river.evaluation.statistics import (
    effect_sizes,
    holm_correction,
    mcnemar_test,
    paired_bootstrap_ci,
    paired_permutation_test,
)


Direction = str
Filter = Callable[[dict[str, Any]], bool]

ENGINEERING_DATA_TOKENS = (
    "authored_unit",
    "illustrative",
    "synthetic",
    "unit_fixture",
    "unit-test",
    "placeholder",
)

BASELINE_ORDER = [
    "naive_rag",
    "stateless_rag",
    "last_write_wins",
    "recency_weighted",
    "legacy_rag",
    "minicheck_no_revision",
    "self_rag",
    "crag",
    "graphrag",
    "graphrag_local",
    "graphrag_global",
    "memgpt",
    "zep",
    "river_d",
    "river_l",
]


class MissingDataError(EvaluationError):
    """Raised when manuscript artifacts cannot be generated without fabricating data."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("missing manuscript data; see missing_data_report.json")


@dataclass(frozen=True)
class MetricColumn:
    name: str
    metric: str
    direction: Direction = "higher"
    ci: bool = True
    label: str | None = None


@dataclass(frozen=True)
class TableSpec:
    table_id: str
    family_number: int
    title: str
    required_columns: tuple[str, ...]
    filters: str
    grouping: tuple[str, ...]
    comparison_mode: str
    directions: dict[str, Direction]
    confidence_intervals: str
    statistical_test_source: str
    caption: str
    footnotes: tuple[str, ...]
    minimum_required_data: str
    # Metadata key this table exists to report on. When no record carries it,
    # the study simply did not collect that kind of data and the table is absent
    # by design rather than missing; a table with no such key is unconditional
    # and its absence is always a failure.
    conditional_on: str | None = None


@dataclass
class TableArtifact:
    spec: TableSpec
    columns: list[str]
    rows: list[dict[str, Any]]
    row_sources: list[list[dict[str, Any]]]


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    title: str
    caption: str
    filters: str
    minimum_required_data: str


@dataclass
class FigureArtifact:
    spec: FigureSpec
    data: list[dict[str, Any]]
    source_records: list[dict[str, Any]]
    x: str
    y: str
    series: str


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and math.isfinite(float(value))


def _mean_or_nan(values: list[Any]) -> float:
    clean = [float(value) for value in values if _finite(value)]
    return mean(clean) if clean else math.nan


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _nested(row: dict[str, Any], *path: str) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _row_label(row: dict[str, Any], field: str) -> str:
    manifest_config = _nested(row, "metadata", "manifest", "config")
    manifest_config = manifest_config if isinstance(manifest_config, dict) else {}
    if field == "condition":
        return str(
            _metadata(row).get("condition")
            or manifest_config.get("condition")
            or _metadata(row).get("ablation")
            or manifest_config.get("ablation")
            or _metadata(row).get("config_name")
            or manifest_config.get("config_name")
            or row.get("baseline", "unknown")
        )
    if field == "mechanism":
        return str(
            _metadata(row).get("mechanism")
            or manifest_config.get("mechanism")
            or _metadata(row).get("ablation")
            or manifest_config.get("ablation")
            or _row_label(row, "condition")
        )
    if field == "threshold":
        return str(_threshold_value(row))
    if field == "generator":
        return str(row.get("generator", "unknown"))
    if field == "audit_group":
        return str(_metadata(row).get("human_audit", {}).get("study_id", row.get("baseline", "unknown")))
    if field == "annotation_source":
        return str(_metadata(row).get("annotation_quality", {}).get("source", row.get("source_dataset", "unknown")))
    return str(row.get(field, "unknown"))


def _threshold_value(row: dict[str, Any]) -> float | None:
    metadata = _metadata(row)
    candidates = [
        metadata.get("threshold"),
        metadata.get("support_threshold"),
        metadata.get("decision_threshold"),
        _nested(row, "metadata", "verification", "verifier", "support_threshold"),
        _nested(row, "metadata", "manifest", "config", "support_threshold"),
    ]
    for candidate in candidates:
        try:
            if candidate is not None and math.isfinite(float(candidate)):
                return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _sort_key(values: tuple[Any, ...]) -> tuple[Any, ...]:
    baseline_rank = {name: index for index, name in enumerate(BASELINE_ORDER)}
    ordered: list[Any] = []
    for value in values:
        if value in baseline_rank:
            ordered.append((0, baseline_rank[value]))
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                ordered.append((2, str(value)))
            else:
                ordered.append((1, numeric) if math.isfinite(numeric) else (2, str(value)))
    return tuple(ordered)


def _records_by_group(records: list[dict[str, Any]], group_by: tuple[str, ...]) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(_row_label(row, field) for field in group_by)].append(row)
    return sorted(grouped.items(), key=lambda item: _sort_key(item[0]))


def _revision_accuracy(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan([revision_correct(row) for row in records])


def _answer_or_revision_accuracy(records: list[dict[str, Any]]) -> float:
    return _mean_or_nan([answer_or_revision_correct(row) for row in records])


def _human_audit_metric(records: list[dict[str, Any]], key: str) -> float:
    values: list[Any] = []
    for row in records:
        audit = _metadata(row).get("human_audit")
        if isinstance(audit, dict):
            values.append(audit.get(key))
    return _mean_or_nan(values)


def _annotation_metric(records: list[dict[str, Any]], key: str) -> float:
    values: list[Any] = []
    for row in records:
        quality = _metadata(row).get("annotation_quality")
        if isinstance(quality, dict):
            values.append(quality.get(key))
    return _mean_or_nan(values)


METRIC_FUNCTIONS: dict[str, Callable[[list[dict[str, Any]]], float]] = {
    **SCALAR_METRICS,
    "revision_accuracy": _revision_accuracy,
    "answer_or_revision_accuracy": _answer_or_revision_accuracy,
    "audit_time_seconds": lambda rows: _human_audit_metric(rows, "audit_time_seconds"),
    "audit_issue_detection_rate": lambda rows: _human_audit_metric(rows, "issue_found"),
    "audit_agreement": lambda rows: _human_audit_metric(rows, "auditor_agreement"),
    "label_agreement": lambda rows: _annotation_metric(rows, "label_agreement"),
    "fleiss_kappa": lambda rows: _annotation_metric(rows, "fleiss_kappa"),
    "adjudication_rate": lambda rows: _annotation_metric(rows, "adjudicated"),
}


def _row_bootstrap_ci(
    records: list[dict[str, Any]],
    metric_name: str,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float] | None:
    value = METRIC_FUNCTIONS[metric_name](records)
    if not _finite(value):
        return None
    units: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        # Keep every seed/run for the same underlying example in one bootstrap
        # cluster. Treating five seeds as five independent test examples makes
        # pooled RIVER-L confidence intervals artificially precise.
        units[(
            row.get("pair_id") or row.get("example_id"),
            row.get("baseline"),
            row.get("comparison_mode", "controlled"),
        )].append(row)
    unit_items = [units[key] for key in sorted(units, key=lambda item: tuple(str(part) for part in item))]
    if len(unit_items) <= 1:
        return float(value), float(value)

    import random

    rng = random.Random(seed)
    estimates: list[float] = []
    for draw in range(resamples):
        sample: list[dict[str, Any]] = []
        for unit_index, unit in enumerate([rng.choice(unit_items) for _ in unit_items]):
            suffix = f"#boot{draw}-{unit_index}"
            for row in unit:
                copied = dict(row)
                if copied.get("pair_id"):
                    copied["pair_id"] = f"{copied['pair_id']}{suffix}"
                copied["example_id"] = f"{copied.get('example_id')}{suffix}-{len(sample)}"
                sample.append(copied)
        estimate = METRIC_FUNCTIONS[metric_name](sample)
        if _finite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return None
    estimates.sort()
    low_index = max(0, min(len(estimates) - 1, int(0.025 * len(estimates))))
    high_index = max(0, min(len(estimates) - 1, int(0.975 * len(estimates)) - 1))
    return estimates[low_index], estimates[high_index]


def _provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prediction_files": sorted({str(row.get("_prediction_file", "<memory>")) for row in records}),
        "run_ids": sorted({str(row.get("run_id")) for row in records if row.get("run_id") is not None}),
        "configuration_hashes": sorted({str(row.get("configuration_hash")) for row in records if row.get("configuration_hash") is not None}),
        "git_commits": sorted({str(row.get("code_commit")) for row in records if row.get("code_commit") is not None}),
        "dataset_hashes": sorted({str(row.get("dataset_hash")) for row in records if row.get("dataset_hash") is not None}),
        "example_ids": sorted({str(row.get("example_id")) for row in records if row.get("example_id") is not None}),
        "n_records": len(records),
    }


def _base_spec(
    table_id: str,
    family_number: int,
    title: str,
    required_columns: tuple[str, ...],
    filters: str,
    grouping: tuple[str, ...],
    comparison_mode: str,
    metric_columns: list[MetricColumn],
    caption: str,
    footnotes: tuple[str, ...],
    minimum_required_data: str,
    statistical_test_source: str = "river.evaluation.statistics",
    conditional_on: str | None = None,
) -> TableSpec:
    directions = {column.name: column.direction for column in metric_columns}
    return TableSpec(
        table_id=table_id,
        family_number=family_number,
        title=title,
        required_columns=required_columns,
        filters=filters,
        grouping=grouping,
        comparison_mode=comparison_mode,
        directions=directions,
        confidence_intervals="95% bootstrap confidence intervals over paired examples where pair_id is available; otherwise over example-level rows.",
        statistical_test_source=statistical_test_source,
        caption=caption,
        footnotes=footnotes,
        minimum_required_data=minimum_required_data,
        conditional_on=conditional_on,
    )


def _metric_columns(*items: tuple[str, str, Direction]) -> list[MetricColumn]:
    return [MetricColumn(name=name, metric=metric, direction=direction) for name, metric, direction in items]


@dataclass(frozen=True)
class Precondition:
    """What a group of records must contain before a metric is defined at all.

    There are two different reasons a cell can hold no number, and collapsing
    them loses the one a reader cares about. A stateless baseline emits no
    revision state, so revision-state metrics have nothing to score: that is a
    property of the system, and the correct cell is "not applicable". A baseline
    that should have recorded its token counts and did not is missing data, and
    that still suppresses the whole table.

    ``requirement`` names what is absent and is printed under the table, so the
    reason an empty cell is empty does not live only in this file.
    """

    requirement: str
    holds: Callable[[list[dict[str, Any]]], bool]


def _any_field(records: list[dict[str, Any]], field: str) -> bool:
    return any(row.get(field) is not None for row in records)


def _any_intervention(records: list[dict[str, Any]], interventions: set[str]) -> bool:
    return any(row.get("intervention") in interventions for row in records)


_NEEDS_STATE = Precondition(
    "a predicted revision state, which this system does not emit",
    lambda records: _any_field(records, "predicted_state"),
)
_NEEDS_CONFIDENCE = Precondition(
    "a confidence score, which this system does not emit",
    lambda records: _any_field(records, "confidence"),
)
_NEEDS_STATE_PROBABILITIES = Precondition(
    "a distribution over revision states, which this system does not emit",
    lambda records: any(
        isinstance(row.get("state_probabilities"), dict) and row["state_probabilities"] for row in records
    ),
)
_NEEDS_ORDER_PAIRS = Precondition(
    "examples whose arrival order was reversed, of which this group contains none",
    lambda records: _any_intervention(records, {"order_reversal", "expired_evidence"}),
)
_NEEDS_STATE_ON_ORDER_PAIRS = Precondition(
    "reversed-order pairs scored by revision correctness, and this system emits no revision state",
    lambda records: _any_intervention(records, {"order_reversal"}) and _any_field(records, "predicted_state"),
)
_NEEDS_LATE_ASSERTION = Precondition(
    "an unsupported late assertion to supersede, of which this group contains none",
    lambda records: any(
        row.get("intervention") == "unsupported_late_assertion"
        and row.get("intervention_hypothesis_id") is not None
        for row in records
    ),
)
_NEEDS_SELECTED_HYPOTHESIS_ON_LATE_ASSERTION = Precondition(
    "a selected hypothesis this system named on at least one late-assertion row, which this system does not emit",
    lambda records: any(
        row.get("intervention") == "unsupported_late_assertion"
        and row.get("intervention_hypothesis_id") is not None
        and (row.get("predicted_hypothesis_id") is not None or row.get("answer_named_hypothesis_id") is not None)
        for row in records
    ),
)
_NEEDS_COST_METADATA = Precondition(
    "recorded generation cost metadata, which this system run did not produce (for example, a --no-generation run)",
    lambda records: any(
        (row.get("metadata") or {}).get("estimated_cost_usd") is not None
        or (row.get("metadata") or {}).get("cost_usd") is not None
        or ((row.get("metadata") or {}).get("native_execution") or {}).get("cost_usd") is not None
        for row in records
    ),
)
_NEEDS_SCOREABLE_ANSWER = Precondition(
    "either a resolvable gold answer or a predicted revision state, and this group has neither",
    lambda records: _any_field(records, "answer_correct") or _any_field(records, "predicted_state"),
)
_NEEDS_STRICT_DIRECT_ANSWER = Precondition(
    "strict direct-answer labels produced under generation contract v2, which this system does not emit",
    lambda records: _any_field(records, "answer_text_correct"),
)
_NEEDS_IN_PROCESS_RUN = Precondition(
    "in-process instrumentation, which a native system run as a subprocess cannot provide",
    lambda records: any(row.get("comparison_mode") != "native" for row in records),
)


# Keyed by the metric function rather than by column name: the reason a metric
# is undefined belongs to the metric, not to whichever table displays it.
METRIC_PRECONDITIONS: dict[str, Precondition] = {
    "answer_text_accuracy": _NEEDS_STRICT_DIRECT_ANSWER,
    "end_to_end_answer_accuracy": _NEEDS_STRICT_DIRECT_ANSWER,
    "revision_accuracy": _NEEDS_STATE,
    "coverage": _NEEDS_STATE,
    "selective_risk": _NEEDS_STATE,
    "abstention_precision": _NEEDS_STATE,
    "abstention_recall": _NEEDS_STATE,
    "abstention_f1": _NEEDS_STATE,
    "conflict_precision": _NEEDS_STATE,
    "conflict_recall": _NEEDS_STATE,
    "conflict_f1": _NEEDS_STATE,
    "conflict_macro_f1": _NEEDS_STATE,
    "conflict_specificity": _NEEDS_STATE,
    "false_conflict_rate": _NEEDS_STATE,
    "expected_calibration_error": _NEEDS_CONFIDENCE,
    "brier_score": _NEEDS_CONFIDENCE,
    "area_under_risk_coverage_curve": _NEEDS_CONFIDENCE,
    "accuracy_at_fixed_coverage": _NEEDS_CONFIDENCE,
    "probability_recency_sensitivity": _NEEDS_STATE_PROBABILITIES,
    "recency_flip_rate": _NEEDS_ORDER_PAIRS,
    "order_reversal_consistency": _NEEDS_ORDER_PAIRS,
    "stable_and_correct": _NEEDS_STATE_ON_ORDER_PAIRS,
    "stable_but_wrong": _NEEDS_STATE_ON_ORDER_PAIRS,
    "flipped_to_correct": _NEEDS_STATE_ON_ORDER_PAIRS,
    "flipped_to_wrong": _NEEDS_STATE_ON_ORDER_PAIRS,
    "unsupported_supersession_rate": _NEEDS_SELECTED_HYPOTHESIS_ON_LATE_ASSERTION,
    "repetition_resistance": _NEEDS_SELECTED_HYPOTHESIS_ON_LATE_ASSERTION,
    "estimated_cost": _NEEDS_COST_METADATA,
    "answer_or_revision_accuracy": _NEEDS_SCOREABLE_ANSWER,
    "revision_latency": _NEEDS_IN_PROCESS_RUN,
    "peak_memory": _NEEDS_IN_PROCESS_RUN,
}


def _filter_records(records: list[dict[str, Any]], predicate: Filter) -> list[dict[str, Any]]:
    return [row for row in records if predicate(row)]


# The conditions that are the system itself rather than a variant of it. A row
# with no ablation label is the system as shipped; "full" and "learned" name the
# unablated reference explicitly.
REFERENCE_CONDITIONS = {None, "full", "learned"}


def _is_ablation_variant(row: dict[str, Any]) -> bool:
    """Whether this row is a deliberately weakened version of a system.

    An ablation is not a system under comparison. Pooling one into the baseline
    it ablates inflates that baseline's sample and drags its metrics toward the
    variant: with four ablations present, RIVER-L's headline row covered nine
    conditions at once and its recency flip rate rose from 0.000 to 0.002 purely
    because the no-temporal-features variant was averaged into it.
    """

    return _metadata(row).get("ablation") not in REFERENCE_CONDITIONS


def _controlled(row: dict[str, Any]) -> bool:
    """Rows eligible for a headline quantitative table.

    The qualitative-only families are dropped here rather than at each call
    site, so a table added later cannot quietly readmit them into an average,
    and so are ablation variants, which belong only to the ablation tables.
    """

    if row.get("comparison_mode", "controlled") != "controlled":
        return False
    if _is_ablation_variant(row):
        return False
    return row.get("intervention") not in QUALITATIVE_ONLY_INTERVENTIONS


def _controlled_including_ablations(row: dict[str, Any]) -> bool:
    """Controlled rows with the ablation variants kept in.

    Only for the two tables whose subject is the ablation. Everywhere else a
    variant would be pooled into the baseline it ablates.
    """

    if row.get("comparison_mode", "controlled") != "controlled":
        return False
    return row.get("intervention") not in QUALITATIVE_ONLY_INTERVENTIONS


def _controlled_including_qualitative(row: dict[str, Any]) -> bool:
    """Controlled rows with the qualitative-only families kept in.

    Only for tables whose subject is one of those families. Such a table reports
    per-family counts beside its rates, so a reader sees the sample it rests on
    instead of an aggregate that hides it.
    """

    return row.get("comparison_mode", "controlled") == "controlled" and not _is_ablation_variant(row)


def _emits_revision_state(row: dict[str, Any]) -> bool:
    """Controlled rows from systems that decide a revision state.

    A retrieval control emits no state, so supersession, conflict and abstention
    are not quantities it can be wrong about. Excluding it names the comparison
    accurately instead of leaving a column with missing rows.
    """

    return _controlled(row) and row.get("predicted_state") is not None


def _native(row: dict[str, Any]) -> bool:
    return row.get("comparison_mode", "controlled") == "native" and not _is_ablation_variant(row)


def _has_claim_labels(row: dict[str, Any]) -> bool:
    for claim in row.get("claim_verification_results") or []:
        if isinstance(claim, dict) and (
            claim.get("gold_label") is not None
            or isinstance(claim.get("metadata"), dict) and claim["metadata"].get("gold_label") is not None
        ):
            return True
    labels = _metadata(row).get("claim_gold_labels") or _metadata(row).get("citation_gold_labels")
    return bool(labels)


def _external_dataset(row: dict[str, Any]) -> bool:
    source = str(row.get("source_dataset", "")).lower()
    dataset = str(row.get("dataset", "")).lower()
    return not any(token in source or token in dataset for token in ENGINEERING_DATA_TOKENS)


def _has_human_audit(row: dict[str, Any]) -> bool:
    return isinstance(_metadata(row).get("human_audit"), dict)


def _has_annotation_quality(row: dict[str, Any]) -> bool:
    return isinstance(_metadata(row).get("annotation_quality"), dict)


def _build_metric_table(
    records: list[dict[str, Any]],
    spec: TableSpec,
    metric_columns: list[MetricColumn],
    *,
    predicate: Filter,
    group_by: tuple[str, ...],
    bootstrap_resamples: int,
    seed: int,
) -> TableArtifact:
    selected = _filter_records(records, predicate)
    if not selected:
        raise MissingDataError({"tables": {spec.table_id: ["no records matched table filters"]}})
    rows: list[dict[str, Any]] = []
    row_sources: list[list[dict[str, Any]]] = []
    columns = [*group_by, "n"]
    for metric in metric_columns:
        columns.extend([metric.name, f"{metric.name}_ci_low", f"{metric.name}_ci_high"])

    for key, items in _records_by_group(selected, group_by):
        row = {field: value for field, value in zip(group_by, key, strict=True)}
        row["n"] = len(items)
        missing: list[str] = []
        not_applicable: dict[str, str] = {}
        for metric in metric_columns:
            precondition = METRIC_PRECONDITIONS.get(metric.metric)
            if precondition is not None and not precondition.holds(items):
                # Undefined rather than unmeasured: leave the cell empty, say why
                # under the table, and do not suppress the table over it.
                not_applicable[metric.name] = precondition.requirement
                row[metric.name] = None
                row[f"{metric.name}_ci_low"] = None
                row[f"{metric.name}_ci_high"] = None
                continue
            value = METRIC_FUNCTIONS[metric.metric](items)
            if not _finite(value):
                missing.append(metric.name)
                row[metric.name] = math.nan
                row[f"{metric.name}_ci_low"] = math.nan
                row[f"{metric.name}_ci_high"] = math.nan
                continue
            row[metric.name] = float(value)
            ci = _row_bootstrap_ci(items, metric.metric, resamples=bootstrap_resamples, seed=seed)
            if ci is None:
                missing.append(f"{metric.name}_ci")
                row[f"{metric.name}_ci_low"] = math.nan
                row[f"{metric.name}_ci_high"] = math.nan
            else:
                row[f"{metric.name}_ci_low"], row[f"{metric.name}_ci_high"] = ci
        if missing:
            row["_missing"] = missing
        if not_applicable:
            row["_not_applicable"] = not_applicable
        rows.append(row)
        row_sources.append(items)

    return TableArtifact(spec=spec, columns=columns, rows=rows, row_sources=row_sources)


def _build_conflict_evolution_table(records: list[dict[str, Any]], bootstrap_resamples: int, seed: int) -> TableArtifact:
    del bootstrap_resamples, seed
    spec = TABLE_SPECS["conflict_versus_temporal_evolution"]
    selected = _filter_records(
        records,
        lambda row: _controlled_including_qualitative(row)
        and row.get("intervention") in {"concurrent_disagreement", "temporal_evolution"},
    )
    if not selected:
        raise MissingDataError({"tables": {spec.table_id: ["no conflict/evolution records matched filters"]}})
    rows: list[dict[str, Any]] = []
    row_sources: list[list[dict[str, Any]]] = []
    for (baseline,), items in _records_by_group(selected, ("baseline",)):
        matrix = conflict_evolution_confusion_matrix(items)
        for gold_label, predictions in matrix.items():
            source_items = [
                row for row in items
                if (row.get("gold_state") if row.get("gold_state") in {"disputed", "evolution"} else "other") == gold_label
            ]
            row = {
                "baseline": baseline,
                "gold_state": gold_label,
                "predicted_disputed": predictions["disputed"],
                "predicted_evolution": predictions["evolution"],
                "predicted_other": predictions["other"],
                "n": sum(predictions.values()),
            }
            rows.append(row)
            row_sources.append(source_items or items)
    return TableArtifact(
        spec=spec,
        columns=["baseline", "gold_state", "predicted_disputed", "predicted_evolution", "predicted_other", "n"],
        rows=rows,
        row_sources=row_sources,
    )


def _build_statistical_table(records: list[dict[str, Any]], bootstrap_resamples: int, seed: int) -> TableArtifact:
    spec = TABLE_SPECS["statistical_comparison_strongest_baseline"]
    selected = _filter_records(records, _controlled)
    grouped = {key[0]: items for key, items in _records_by_group(selected, ("baseline",))}
    # The comparison below is a difference in revision accuracy, so a baseline
    # that emits no revision state cannot serve as the reference: picking it on
    # answer accuracy alone would yield a row of NAs instead of a comparison.
    non_river = {
        baseline: items
        for baseline, items in grouped.items()
        if not str(baseline).startswith("river_")
        and any(revision_correct(row) is not None for row in items)
    }
    river = {baseline: items for baseline, items in grouped.items() if str(baseline).startswith("river_")}
    if not non_river or not river:
        raise MissingDataError({ "tables": {spec.table_id: ["requires at least one RIVER baseline and one non-RIVER baseline that emits a revision state"]}})
    strongest = max(non_river, key=lambda baseline: (_answer_or_revision_accuracy(non_river[baseline]), -SCALAR_METRICS["recency_flip_rate"](non_river[baseline]) if _finite(SCALAR_METRICS["recency_flip_rate"](non_river[baseline])) else -999))
    rows: list[dict[str, Any]] = []
    row_sources: list[list[dict[str, Any]]] = []
    for candidate in sorted(river, key=lambda name: BASELINE_ORDER.index(name) if name in BASELINE_ORDER else 999):
        candidate_rows, reference_rows = _align_by_example(selected, candidate, strongest)
        candidate_correct = [revision_correct(row) for row in candidate_rows]
        reference_correct = [revision_correct(row) for row in reference_rows]
        ci = paired_bootstrap_ci(candidate_correct, reference_correct, n_resamples=bootstrap_resamples, seed=seed)
        permutation = paired_permutation_test(candidate_correct, reference_correct, seed=seed)
        mcnemar = mcnemar_test(candidate_correct, reference_correct)
        effects = effect_sizes(candidate_correct, reference_correct)
        rows.append({
            "candidate": candidate,
            "reference": strongest,
            "metric": "revision_accuracy_difference",
            "estimate": ci["estimate"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "mcnemar_p": mcnemar["p_value"],
            "mcnemar_p_holm": math.nan,
            "permutation_p": permutation["p_value"],
            "permutation_p_holm": math.nan,
            "cohen_dz": effects["cohen_dz"],
            "n": ci["n"],
        })
        row_sources.append([*candidate_rows, *reference_rows])
    mcnemar_adjusted = holm_correction({row["candidate"]: row["mcnemar_p"] for row in rows})
    permutation_adjusted = holm_correction({row["candidate"]: row["permutation_p"] for row in rows})
    for row in rows:
        row["mcnemar_p_holm"] = mcnemar_adjusted[row["candidate"]]
        row["permutation_p_holm"] = permutation_adjusted[row["candidate"]]
    return TableArtifact(
        spec=spec,
        columns=[
            "candidate",
            "reference",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "mcnemar_p",
            "mcnemar_p_holm",
            "permutation_p",
            "permutation_p_holm",
            "cohen_dz",
            "n",
        ],
        rows=rows,
        row_sources=row_sources,
    )


def _align_by_example(records: list[dict[str, Any]], left: str, right: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left_map = {
        (row.get("example_id"), row.get("seed")): row
        for row in records
        if row.get("baseline") == left
    }
    right_map = {
        (row.get("example_id"), row.get("seed")): row
        for row in records
        if row.get("baseline") == right
    }
    shared = sorted(set(left_map) & set(right_map))
    if not shared:
        raise MissingDataError({"tables": {"statistical_comparison_strongest_baseline": [f"no shared examples for {left} versus {right}"]}})
    return [left_map[key] for key in shared], [right_map[key] for key in shared]


def _table_specs() -> dict[str, TableSpec]:
    specs: dict[str, TableSpec] = {}

    def add(spec: TableSpec) -> None:
        specs[spec.table_id] = spec

    add(_base_spec(
        "revision_state_comparison", 1, "Revision-state comparison",
        ("baseline", "comparison_mode", "gold_state", "predicted_state"),
        "comparison_mode == controlled; rows from systems that emit a revision state", ("baseline",), "controlled",
        _metric_columns(
            ("conflict_macro_f1", "conflict_macro_f1", "higher"),
            ("abstention_f1", "abstention_f1", "higher"),
        ),
        "Conflict detection and abstention among systems that decide a revision state.",
        ("Systems emitting no revision state are absent by construction, not by omission.",),
        "At least one controlled row carrying a predicted state.",
    ))

    add(_base_spec(
        "overall_controlled_baseline_comparison", 1, "Overall controlled baseline comparison",
        ("baseline", "comparison_mode", "gold_state", "predicted_state", "gold_hypothesis_id", "predicted_hypothesis_id"),
        "comparison_mode == controlled; all labelled test rows", ("baseline",), "controlled",
        _metric_columns(
            ("answer_accuracy", "answer_text_accuracy", "higher"),
            ("rfr", "recency_flip_rate", "lower"),
            ("usr", "unsupported_supersession_rate", "lower"),
        ),
        "Overall controlled comparison across normalized RIVER-Bench prediction records.",
        ("Controlled rows share evidence and budgets where technically valid.", "Lower is better for RFR, USR, and ECE."),
        "At least two controlled baselines with complete primary labels.",
    ))
    add(_base_spec(
        "results_by_intervention", 2, "Results by intervention",
        ("intervention", "baseline", "comparison_mode", "gold_state", "predicted_state"), "controlled rows grouped by intervention and baseline",
        ("intervention", "baseline"), "controlled",
        _metric_columns(("accuracy", "revision_accuracy", "higher"), ("coverage", "coverage", "higher")),
        "Per-intervention revision accuracy and coverage.",
        ("Each intervention row is computed only from examples carrying that intervention label.",),
        "At least one labelled row for each reported intervention-baseline group.",
    ))
    add(_base_spec(
        "paired_recency_decomposition", 3, "Paired recency decomposition",
        ("pair_id", "intervention", "baseline", "comparison_mode", "predicted_state", "predicted_hypothesis_id"),
        "controlled order_reversal pairs only", ("baseline",), "controlled",
        _metric_columns(
            ("rfr", "recency_flip_rate", "lower"),
            ("prs_jsd", "probability_recency_sensitivity", "lower"),
            ("order_consistency", "order_reversal_consistency", "higher"),
            ("stable_correct", "stable_and_correct", "higher"),
            ("stable_wrong", "stable_but_wrong", "lower"),
            ("flip_to_correct", "flipped_to_correct", "higher"),
            ("flip_to_wrong", "flipped_to_wrong", "lower"),
        ),
        "Arrival-order sensitivity decomposed on paired order-reversal examples.",
        ("Pair IDs are mandatory; rows without complete pairs are rejected.",),
        "Complete order_reversal pairs with state and hypothesis predictions.",
    ))
    add(_base_spec(
        "main_river_ablation", 4, "Main RIVER ablation",
        ("baseline", "comparison_mode", "metadata"), "controlled RIVER and ablation-labelled rows",
        ("condition",), "controlled",
        _metric_columns(("state_accuracy", "revision_accuracy", "higher"), ("rfr", "recency_flip_rate", "lower"), ("usr", "unsupported_supersession_rate", "lower"), ("abstention_f1", "abstention_f1", "higher")),
        "Main ablation study for RIVER mechanisms.",
        (
            "Condition comes from metadata.condition, metadata.ablation, metadata.config_name, or baseline.",
            "State accuracy is reported because the other three columns are computed on the "
            "order-reversal, unsupported-late-assertion and NEI families only. An ablation of "
            "the temporal mechanism acts on expired evidence, which none of those look at, so "
            "without this column it would read as no effect while changing decisions.",
        ),
        "At least two RIVER/ablation conditions with complete primary labels.",
    ))
    add(_base_spec(
        "mechanism_to_outcome_ablation", 5, "Mechanism-to-outcome ablation",
        ("metadata", "baseline", "comparison_mode"), "controlled rows with mechanism or ablation metadata",
        ("mechanism",), "controlled",
        _metric_columns(("rfr", "recency_flip_rate", "lower"), ("conflict_f1", "conflict_macro_f1", "higher"), ("abstention_f1", "abstention_f1", "higher"), ("brier", "brier_score", "lower")),
        "Mechanism-level ablations mapped to outcomes.",
        ("Mechanism labels are read from metadata.mechanism or metadata.ablation.",),
        "At least two mechanism-labelled groups.",
    ))
    add(_base_spec(
        "river_d_vs_river_l", 6, "RIVER-D versus RIVER-L",
        ("baseline", "comparison_mode"), "controlled rows where baseline is river_d or river_l",
        ("baseline",), "controlled",
        _metric_columns(("rfr", "recency_flip_rate", "lower"), ("ece", "expected_calibration_error", "lower"), ("cost", "estimated_cost", "lower")),
        "Direct deterministic-versus-learned RIVER comparison.",
        ("RIVER-L rows require a frozen checkpoint recorded in prediction metadata.",),
        "Both river_d and river_l controlled predictions.",
    ))
    add(_base_spec(
        "compliance_support_correctness_decomposition", 8, "Compliance/support/correctness decomposition",
        ("citation_supported", "answer_text_correct", "claim_verification_results", "comparison_mode"), "rows with citation and strict direct-answer labels",
        ("comparison_mode", "baseline"), "controlled and native rows reported separately",
        _metric_columns(("supported_text_correct", "supported_and_text_correct", "higher"), ("compliant_unsupported", "compliant_but_unsupported", "lower"), ("supported_text_wrong", "supported_but_text_wrong", "lower")),
        "Exploratory decomposition of strict direct-answer correctness and predicted citation support.",
        ("This is a system-internal diagnostic, not gold citation-quality evaluation. Rows missing either field are excluded.",),
        "Rows with citation_supported, strict answer_text_correct, and claim citation compliance labels.",
        conditional_on="claim_verification_results",
    ))
    add(_base_spec(
        "conflict_detection_negative_controls", 9, "Conflict detection with negative controls",
        ("gold_state", "predicted_state", "intervention", "comparison_mode"), "controlled conflict and non-conflict controls",
        ("baseline",), "controlled",
        _metric_columns(("precision", "conflict_precision", "higher"), ("recall", "conflict_recall", "higher"), ("macro_f1", "conflict_macro_f1", "higher"), ("specificity", "conflict_specificity", "higher"), ("false_conflict", "false_conflict_rate", "lower")),
        "Conflict detection including negative controls.",
        ("Temporal-evolution and non-conflict interventions serve as negative controls.",),
        "At least one disputed and one non-disputed labelled row per reported group.",
    ))
    add(_base_spec(
        "conflict_versus_temporal_evolution", 10, "Conflict versus temporal evolution",
        ("gold_state", "predicted_state", "intervention", "baseline", "comparison_mode"), "controlled concurrent_disagreement and temporal_evolution rows",
        ("baseline", "gold_state"), "controlled",
        [], "Confusion matrix separating concurrent disagreement from temporal evolution.",
        ("Cells are raw counts, not rates.",), "Conflict and evolution labelled rows.",
    ))
    add(_base_spec(
        "nei_selective_abstention", 11, "NEI and selective abstention",
        ("gold_state", "predicted_state", "confidence", "comparison_mode"), "controlled insufficient_evidence_nei and evidence_removal rows",
        ("baseline",), "controlled",
        _metric_columns(("precision", "abstention_precision", "higher"), ("recall", "abstention_recall", "higher"), ("f1", "abstention_f1", "higher"), ("coverage", "coverage", "higher"), ("selective_risk", "selective_risk", "lower"), ("aurc", "area_under_risk_coverage_curve", "lower"), ("acc_at_80cov", "accuracy_at_fixed_coverage", "higher")),
        "Selective abstention on NEI and evidence-removal examples.",
        ("Risk-coverage metrics require confidence values and correctness labels.",),
        "Rows with abstention labels, confidence, and correctness.",
    ))
    add(_base_spec(
        "external_dataset_generalization", 12, "External-dataset generalization",
        ("source_dataset", "baseline", "comparison_mode"), "non-engineering external source datasets",
        ("comparison_mode", "source_dataset", "baseline"), "controlled and native rows reported separately",
        _metric_columns(("accuracy", "answer_or_revision_accuracy", "higher"), ("rfr", "recency_flip_rate", "lower"), ("abstention_f1", "abstention_f1", "higher")),
        "Generalization across external datasets.",
        ("Authored unit fixtures and synthetic sources are refused for this table.",),
        "At least one non-engineering source_dataset with complete labels.",
    ))
    add(_base_spec(
        "cross_model_reproducibility", 13, "Cross-model reproducibility",
        ("generator", "baseline", "comparison_mode"), "rows grouped by generator/model and baseline",
        ("comparison_mode", "generator", "baseline"), "controlled and native rows reported separately",
        _metric_columns(("accuracy", "answer_text_accuracy", "higher"), ("rfr", "recency_flip_rate", "lower"), ("usr", "unsupported_supersession_rate", "lower")),
        "Reproducibility across generator/model settings.",
        ("Generator is taken from PredictionRecord.generator.",),
        "At least two generator labels with complete labels.",
    ))
    add(TableSpec(
        table_id="statistical_comparison_strongest_baseline",
        family_number=14,
        title="Statistical comparison with strongest baseline",
        required_columns=("example_id", "baseline", "seed"),
        filters="controlled rows aligned by example_id and seed",
        grouping=("candidate", "reference"),
        comparison_mode="controlled",
        directions={"estimate": "higher", "mcnemar_p": "lower", "mcnemar_p_holm": "lower", "permutation_p": "lower", "permutation_p_holm": "lower", "cohen_dz": "higher"},
        confidence_intervals="Paired bootstrap confidence intervals over aligned examples.",
        statistical_test_source="river.evaluation.statistics",
        caption="Paired statistical comparison against the strongest non-RIVER baseline.",
        footnotes=("The strongest baseline is selected by answer-or-revision accuracy, excluding RIVER variants.",),
        minimum_required_data="At least one RIVER baseline, one non-RIVER baseline, and shared example IDs.",
    ))
    add(_base_spec(
        "threshold_sensitivity", 15, "Threshold sensitivity",
        ("metadata", "baseline", "comparison_mode"), "rows with threshold metadata",
        ("comparison_mode", "baseline", "threshold"), "controlled and native rows reported separately",
        _metric_columns(("rfr", "recency_flip_rate", "lower"), ("abstention_f1", "abstention_f1", "higher"), ("ece", "expected_calibration_error", "lower")),
        "Sensitivity of primary outcomes to configured thresholds.",
        ("Threshold is read from metadata.threshold, support_threshold, decision_threshold, or verifier support_threshold.",),
        "At least two threshold values.",
    ))
    add(_base_spec(
        "computational_cost", 16, "Computational cost",
        ("latency_ms", "input_tokens", "output_tokens", "metadata", "baseline", "comparison_mode"), "rows with runtime telemetry",
        ("comparison_mode", "baseline"), "controlled and native rows reported separately",
        _metric_columns(("latency_ms", "end_to_end_latency", "lower"), ("revision_latency_ms", "revision_latency", "lower"), ("tokens", "tokens", "lower"), ("model_calls", "model_calls", "lower"), ("nli_calls", "nli_calls", "lower"), ("peak_memory_mb", "peak_memory", "lower"), ("estimated_cost", "estimated_cost", "lower")),
        "Computational cost and telemetry.",
        ("Unavailable telemetry is reported as missing rather than zero.",),
        "Runtime, token, call, memory, or cost telemetry for each reported group.",
    ))
    add(_base_spec(
        "human_auditing_study", 17, "Human auditing study",
        ("metadata", "baseline", "comparison_mode"), "rows with metadata.human_audit",
        ("comparison_mode", "audit_group", "baseline"), "controlled and native rows reported separately",
        _metric_columns(("audit_time_seconds", "audit_time_seconds", "lower"), ("issue_detection_rate", "audit_issue_detection_rate", "higher"), ("auditor_agreement", "audit_agreement", "higher")),
        "Human audit study outcomes.",
        ("This table is absent unless human-audit metadata are present.",),
        "Rows with metadata.human_audit audit_time_seconds, issue_found, and auditor_agreement.",
        conditional_on="human_audit",
    ))
    add(_base_spec(
        "annotation_quality", 18, "Annotation quality",
        ("metadata", "comparison_mode"), "rows with metadata.annotation_quality",
        ("comparison_mode", "annotation_source"), "controlled and native rows reported separately",
        _metric_columns(("label_agreement", "label_agreement", "higher"), ("fleiss_kappa", "fleiss_kappa", "higher"), ("adjudication_rate", "adjudication_rate", "lower")),
        "Annotation quality and adjudication statistics.",
        ("This table is absent unless annotation-quality metadata are present.",),
        "Rows with metadata.annotation_quality label_agreement, fleiss_kappa, and adjudicated.",
        conditional_on="annotation_quality",
    ))
    return specs


TABLE_SPECS = _table_specs()


TABLE_BUILDERS: dict[str, Callable[[list[dict[str, Any]], int, int], TableArtifact]] = {
    "conflict_versus_temporal_evolution": _build_conflict_evolution_table,
    "statistical_comparison_strongest_baseline": _build_statistical_table,
}


TABLE_METRICS: dict[str, tuple[list[MetricColumn], Filter, tuple[str, ...]]] = {
    "overall_controlled_baseline_comparison": (
        _metric_columns(("answer_accuracy", "answer_text_accuracy", "higher"), ("rfr", "recency_flip_rate", "lower"), ("usr", "unsupported_supersession_rate", "lower")),
        _controlled, ("baseline",),
    ),
    "revision_state_comparison": (
        _metric_columns(("conflict_macro_f1", "conflict_macro_f1", "higher"), ("abstention_f1", "abstention_f1", "higher")),
        _emits_revision_state, ("baseline",),
    ),
    "results_by_intervention": (
        _metric_columns(("accuracy", "revision_accuracy", "higher"), ("coverage", "coverage", "higher")),
        _controlled_including_qualitative, ("intervention", "baseline"),
    ),
    "paired_recency_decomposition": (
        _metric_columns(("rfr", "recency_flip_rate", "lower"), ("prs_jsd", "probability_recency_sensitivity", "lower"), ("order_consistency", "order_reversal_consistency", "higher"), ("stable_correct", "stable_and_correct", "higher"), ("stable_wrong", "stable_but_wrong", "lower"), ("flip_to_correct", "flipped_to_correct", "higher"), ("flip_to_wrong", "flipped_to_wrong", "lower")),
        lambda row: _controlled(row) and row.get("intervention") == "order_reversal", ("baseline",),
    ),
    "main_river_ablation": (
        _metric_columns(("state_accuracy", "revision_accuracy", "higher"), ("rfr", "recency_flip_rate", "lower"), ("usr", "unsupported_supersession_rate", "lower"), ("abstention_f1", "abstention_f1", "higher")),
        lambda row: _controlled_including_ablations(row) and (str(row.get("baseline", "")).startswith("river_") or _metadata(row).get("ablation") or _metadata(row).get("condition")),
        ("condition",),
    ),
    "mechanism_to_outcome_ablation": (
        _metric_columns(("rfr", "recency_flip_rate", "lower"), ("conflict_f1", "conflict_macro_f1", "higher"), ("abstention_f1", "abstention_f1", "higher"), ("brier", "brier_score", "lower")),
        lambda row: _controlled_including_ablations(row) and (_metadata(row).get("mechanism") or _metadata(row).get("ablation")),
        ("mechanism",),
    ),
    "river_d_vs_river_l": (
        _metric_columns(("rfr", "recency_flip_rate", "lower"), ("ece", "expected_calibration_error", "lower"), ("cost", "estimated_cost", "lower")),
        lambda row: _controlled(row) and row.get("baseline") in {"river_d", "river_l"}, ("baseline",),
    ),
    "compliance_support_correctness_decomposition": (
        _metric_columns(("supported_text_correct", "supported_and_text_correct", "higher"), ("compliant_unsupported", "compliant_but_unsupported", "lower"), ("supported_text_wrong", "supported_but_text_wrong", "lower")),
        lambda row: row.get("citation_supported") is not None and row.get("answer_text_correct") is not None and bool(row.get("claim_verification_results")),
        ("comparison_mode", "baseline"),
    ),
    "conflict_detection_negative_controls": (
        _metric_columns(("precision", "conflict_precision", "higher"), ("recall", "conflict_recall", "higher"), ("macro_f1", "conflict_macro_f1", "higher"), ("specificity", "conflict_specificity", "higher"), ("false_conflict", "false_conflict_rate", "lower")),
        lambda row: _controlled(row) and row.get("intervention") in {"concurrent_disagreement", "temporal_evolution", "order_reversal", "unsupported_late_assertion"},
        ("baseline",),
    ),
    "nei_selective_abstention": (
        _metric_columns(("precision", "abstention_precision", "higher"), ("recall", "abstention_recall", "higher"), ("f1", "abstention_f1", "higher"), ("coverage", "coverage", "higher"), ("selective_risk", "selective_risk", "lower"), ("aurc", "area_under_risk_coverage_curve", "lower"), ("acc_at_80cov", "accuracy_at_fixed_coverage", "higher")),
        lambda row: _controlled(row) and row.get("intervention") in {"insufficient_evidence_nei", "evidence_removal"},
        ("baseline",),
    ),
    "external_dataset_generalization": (
        _metric_columns(("accuracy", "answer_or_revision_accuracy", "higher"), ("rfr", "recency_flip_rate", "lower"), ("abstention_f1", "abstention_f1", "higher")),
        _external_dataset, ("comparison_mode", "source_dataset", "baseline"),
    ),
    "cross_model_reproducibility": (
        _metric_columns(("accuracy", "answer_text_accuracy", "higher"), ("rfr", "recency_flip_rate", "lower"), ("usr", "unsupported_supersession_rate", "lower")),
        lambda row: row.get("generator") is not None, ("comparison_mode", "generator", "baseline"),
    ),
    "threshold_sensitivity": (
        _metric_columns(("rfr", "recency_flip_rate", "lower"), ("abstention_f1", "abstention_f1", "higher"), ("ece", "expected_calibration_error", "lower")),
        lambda row: _threshold_value(row) is not None, ("comparison_mode", "baseline", "threshold"),
    ),
    "computational_cost": (
        _metric_columns(("latency_ms", "end_to_end_latency", "lower"), ("revision_latency_ms", "revision_latency", "lower"), ("tokens", "tokens", "lower"), ("model_calls", "model_calls", "lower"), ("nli_calls", "nli_calls", "lower"), ("peak_memory_mb", "peak_memory", "lower"), ("estimated_cost", "estimated_cost", "lower")),
        lambda row: True, ("comparison_mode", "baseline"),
    ),
    "human_auditing_study": (
        _metric_columns(("audit_time_seconds", "audit_time_seconds", "lower"), ("issue_detection_rate", "audit_issue_detection_rate", "higher"), ("auditor_agreement", "audit_agreement", "higher")),
        _has_human_audit, ("comparison_mode", "audit_group", "baseline"),
    ),
    "annotation_quality": (
        _metric_columns(("label_agreement", "label_agreement", "higher"), ("fleiss_kappa", "fleiss_kappa", "higher"), ("adjudication_rate", "adjudication_rate", "lower")),
        _has_annotation_quality, ("comparison_mode", "annotation_source"),
    ),
}


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
            if item.get("_type") == "run_config":
                continue
            item["_prediction_file"] = str(path)
            records.append(item)
    return records


def _validate_no_engineering_or_placeholder(records: list[dict[str, Any]], *, allow_engineering_data: bool) -> list[str]:
    if allow_engineering_data:
        return []
    failures: list[str] = []
    for row in records:
        searchable = " ".join(
            str(value).lower()
            for value in [
                row.get("dataset"),
                row.get("source_dataset"),
                _metadata(row).get("dataset_note"),
                _metadata(row).get("fixture_type"),
            ]
            if value is not None
        )
        if any(token in searchable for token in ENGINEERING_DATA_TOKENS):
            failures.append(str(row.get("example_id", "<unknown>")))
    return sorted(set(failures))


def _validate_required_input_columns(records: list[dict[str, Any]], spec: TableSpec) -> list[str]:
    missing = []
    for column in spec.required_columns:
        if column == "metadata":
            if not any(isinstance(row.get("metadata"), dict) for row in records):
                missing.append(column)
            continue
        if not any(column in row and row.get(column) is not None for row in records):
            missing.append(column)
    return missing


def _validate_table(artifact: TableArtifact) -> list[str]:
    failures: list[str] = []
    if not artifact.rows:
        failures.append("no output rows")
    for row_index, row in enumerate(artifact.rows):
        missing = row.get("_missing") or []
        if missing:
            failures.append(f"row {row_index} missing required metrics: {', '.join(missing)}")
        for column in artifact.columns:
            value = row.get(column)
            if isinstance(value, float) and not math.isfinite(value):
                failures.append(f"row {row_index} column {column} is missing or non-finite")
    return failures


def _build_table(records: list[dict[str, Any]], table_id: str, *, bootstrap_resamples: int, seed: int) -> TableArtifact:
    if table_id in TABLE_BUILDERS:
        return TABLE_BUILDERS[table_id](records, bootstrap_resamples, seed)
    spec = TABLE_SPECS[table_id]
    metric_columns, predicate, group_by = TABLE_METRICS[table_id]
    return _build_metric_table(records, spec, metric_columns, predicate=predicate, group_by=group_by, bootstrap_resamples=bootstrap_resamples, seed=seed)


def _best_columns(artifact: TableArtifact) -> dict[str, set[int]]:
    best: dict[str, set[int]] = {}
    for column, direction in artifact.spec.directions.items():
        values = [
            (index, row.get(column))
            for index, row in enumerate(artifact.rows)
            if _finite(row.get(column))
        ]
        if not values or direction == "neutral":
            continue
        target = max(value for _, value in values) if direction == "higher" else min(value for _, value in values)
        best[column] = {index for index, value in values if float(value) == float(target)}
    return best


def _format_number(value: float) -> str:
    return f"{value:.3f}"


def _format_cell(value: Any, *, bold: bool = False, latex: bool = False) -> str:
    if value is None:
        text = "NA"
    elif isinstance(value, float):
        if not math.isfinite(value):
            text = "NA"
        else:
            text = _format_number(value)
    else:
        text = str(value)
    if latex:
        text = latex_escape(text)
        return f"\\textbf{{{text}}}" if bold else text
    return f"**{text}**" if bold else text


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON output contains NaN or Infinity")
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _public_rows(artifact: TableArtifact) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in artifact.columns} for row in artifact.rows]


def _cell_provenance(artifact: TableArtifact) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for row_index, row in enumerate(artifact.rows):
        row_key = "|".join(str(row.get(column, row_index)) for column in artifact.columns[: max(1, len(artifact.spec.grouping))])
        source_records = artifact.row_sources[row_index]
        for column in artifact.columns:
            cells[f"{row_key}:{column}"] = _provenance(source_records)
    return cells


# Above this many affected groups a note counts them instead of listing them.
# The row-level detail is always available in the table's JSON.
_NOTE_LIST_LIMIT = 6


def _not_applicable_notes(artifact: TableArtifact) -> list[str]:
    """One note per column and reason, naming the rows it left empty."""

    grouping = artifact.spec.grouping or tuple(artifact.columns[:1])
    by_column: dict[tuple[str, str], list[str]] = {}
    for row in artifact.rows:
        entries = row.get("_not_applicable") or {}
        if not entries:
            continue
        label = " / ".join(str(row.get(field)) for field in grouping if row.get(field) is not None)
        for column, requirement in entries.items():
            by_column.setdefault((column, requirement), []).append(label)

    total = len(artifact.rows)
    notes: list[str] = []
    for (column, requirement), labels in sorted(by_column.items()):
        unique = sorted(set(labels))
        if len(unique) == total:
            where = "every row"
        elif len(unique) > _NOTE_LIST_LIMIT:
            where = f"{len(unique)} of {total} rows"
        else:
            where = ", ".join(unique)
        notes.append(f"{column} is not applicable for {where}: it needs {requirement}.")
    return notes


def _write_table(artifact: TableArtifact, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    public_rows = _public_rows(artifact)
    best = _best_columns(artifact)
    footnotes = (*artifact.spec.footnotes, *_not_applicable_notes(artifact))
    stem = f"table_{artifact.spec.family_number:02d}_{artifact.spec.table_id}"

    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=artifact.columns)
        writer.writeheader()
        writer.writerows(public_rows)

    json_payload = {
        "spec": asdict(artifact.spec),
        "columns": artifact.columns,
        "rows": public_rows,
        # Machine-readable counterpart to the footnotes: which empty cells are
        # undefined by construction, and what each of them would have needed.
        "not_applicable": {
            "|".join(str(row.get(field)) for field in (artifact.spec.grouping or artifact.columns[:1])): row["_not_applicable"]
            for row in artifact.rows
            if row.get("_not_applicable")
        },
    }
    (output / f"{stem}.json").write_text(json.dumps(_json_safe(json_payload), indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [
        f"### Table {artifact.spec.family_number}. {artifact.spec.title}",
        "",
        artifact.spec.caption,
        "",
        "| " + " | ".join(artifact.columns) + " |",
        "| " + " | ".join("---" for _ in artifact.columns) + " |",
    ]
    for row_index, row in enumerate(public_rows):
        md_lines.append("| " + " | ".join(
            _format_cell(row.get(column), bold=row_index in best.get(column, set()))
            for column in artifact.columns
        ) + " |")
    md_lines.extend(["", *[f"- {note}" for note in footnotes]])
    (output / f"{stem}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latex_lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{{latex_escape(artifact.spec.caption)}}}",
        r"\begin{tabular}{" + "l" * len(artifact.columns) + "}",
        " & ".join(latex_escape(column) for column in artifact.columns) + r" \\ \hline",
    ]
    for row_index, row in enumerate(public_rows):
        latex_lines.append(" & ".join(
            _format_cell(row.get(column), bold=row_index in best.get(column, set()), latex=True)
            for column in artifact.columns
        ) + r" \\")
    latex_lines.extend([r"\end{tabular}", *[rf"\footnotesize{{{latex_escape(note)}}}" for note in footnotes], r"\end{table}"])
    (output / f"{stem}.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    return {
        "spec": asdict(artifact.spec),
        "columns": artifact.columns,
        "rows": public_rows,
    }


def _group_records(records: list[dict[str, Any]], fields: tuple[str, ...]) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    return _records_by_group(records, fields)


def _figure_risk_coverage(records: list[dict[str, Any]]) -> FigureArtifact:
    spec = FIGURE_SPECS["risk_coverage"]
    records = _filter_records(records, _controlled)
    data: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for (baseline,), items in _group_records(records, ("baseline",)):
        curve = risk_coverage_curve(items)
        for point in curve:
            data.append({"baseline": baseline, **point})
            sources.extend(items)
    if not data:
        raise MissingDataError({"figures": {spec.figure_id: ["requires confidence and correctness labels"]}})
    return FigureArtifact(spec, data, sources, "coverage", "risk", "baseline")


def _figure_threshold_sensitivity(records: list[dict[str, Any]]) -> FigureArtifact:
    spec = FIGURE_SPECS["threshold_sensitivity"]
    records = _filter_records(records, _controlled)
    data = []
    sources: list[dict[str, Any]] = []
    for (baseline, threshold), items in _group_records([row for row in records if _threshold_value(row) is not None], ("baseline", "threshold")):
        accuracy = SCALAR_METRICS["answer_accuracy"](items)
        if _finite(accuracy):
            data.append({"baseline": baseline, "threshold": float(threshold), "accuracy": accuracy})
            sources.extend(items)
    if len({item["threshold"] for item in data}) < 2:
        raise MissingDataError({"figures": {spec.figure_id: ["requires at least two threshold values"]}})
    return FigureArtifact(spec, data, sources, "threshold", "accuracy", "baseline")


def _figure_calibration(records: list[dict[str, Any]]) -> FigureArtifact:
    spec = FIGURE_SPECS["calibration"]
    records = _filter_records(records, _controlled)
    data = []
    sources: list[dict[str, Any]] = []
    for (baseline,), items in _group_records(records, ("baseline",)):
        for item in reliability_bins(items):
            if item["n"]:
                data.append({"baseline": baseline, "confidence": item["confidence"], "accuracy": item["accuracy"], "n": item["n"]})
                sources.extend(items)
    if not data:
        raise MissingDataError({"figures": {spec.figure_id: ["requires confidence and correctness labels"]}})
    return FigureArtifact(spec, data, sources, "confidence", "accuracy", "baseline")


def _figure_intervention_effects(records: list[dict[str, Any]]) -> FigureArtifact:
    spec = FIGURE_SPECS["per_intervention_effects"]
    records = _filter_records(records, _controlled)
    data = []
    sources: list[dict[str, Any]] = []
    for (intervention, baseline), items in _group_records(records, ("intervention", "baseline")):
        accuracy = _revision_accuracy(items)
        if _finite(accuracy):
            data.append({"intervention": intervention, "baseline": baseline, "accuracy": accuracy})
            sources.extend(items)
    if not data:
        raise MissingDataError({"figures": {spec.figure_id: ["requires intervention labels and revision correctness"]}})
    return FigureArtifact(spec, data, sources, "intervention", "accuracy", "baseline")


FIGURE_SPECS = {
    "risk_coverage": FigureSpec("risk_coverage", "Risk-coverage", "Selective risk as coverage increases.", "controlled confidence-ranked predictions", "controlled confidence and correctness labels"),
    "threshold_sensitivity": FigureSpec("threshold_sensitivity", "Threshold sensitivity", "Answer accuracy as a function of decision threshold.", "controlled threshold-labelled predictions", "at least two controlled threshold values"),
    "calibration": FigureSpec("calibration", "Calibration", "Reliability curve by baseline.", "controlled confidence-labelled predictions", "controlled confidence and correctness labels"),
    "per_intervention_effects": FigureSpec("per_intervention_effects", "Per-intervention effects", "Revision accuracy by intervention.", "controlled intervention-labelled predictions", "controlled labelled intervention rows"),
}


FIGURE_BUILDERS = {
    "risk_coverage": _figure_risk_coverage,
    "threshold_sensitivity": _figure_threshold_sensitivity,
    "calibration": _figure_calibration,
    "per_intervention_effects": _figure_intervention_effects,
}


def _write_figure(artifact: FigureArtifact, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    json_payload = {"spec": asdict(artifact.spec), "data": artifact.data}
    (output / f"figure_{artifact.spec.figure_id}.json").write_text(json.dumps(_json_safe(json_payload), indent=2, sort_keys=True), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is installed in CI, but keep failure explicit.
        raise MissingDataError({"figures": {artifact.spec.figure_id: [f"matplotlib unavailable: {exc}"]}}) from exc

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in artifact.data:
        groups[str(row[artifact.series])].append(row)
    for label, rows in sorted(groups.items()):
        xs = [row[artifact.x] for row in rows]
        ys = [row[artifact.y] for row in rows]
        if artifact.x == "intervention":
            positions = list(range(len(xs)))
            ax.plot(positions, ys, marker="o", label=label)
            ax.set_xticks(positions, xs, rotation=45, ha="right")
        else:
            ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel(artifact.x)
    ax.set_ylabel(artifact.y)
    ax.set_title(artifact.spec.title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / f"figure_{artifact.spec.figure_id}.png")
    plt.close(fig)
    return json_payload


def _missing_report(tables: dict[str, list[str]], figures: dict[str, list[str]], engineering_rows: list[str]) -> dict[str, Any]:
    return {
        "status": "missing_data",
        "tables": tables,
        "figures": figures,
        "engineering_or_placeholder_rows": engineering_rows,
        "message": "Manuscript artifacts were not generated because required labelled data were missing or unsafe for final-looking tables.",
    }


def generate_artifacts(
    records: list[dict[str, Any]],
    output: Path,
    *,
    requested_tables: list[str] | None = None,
    requested_figures: list[str] | None = None,
    bootstrap_resamples: int = 500,
    seed: int = 13,
    allow_engineering_data: bool = False,
) -> dict[str, Any]:
    if not records:
        report = _missing_report({"all": ["no prediction records found"]}, {}, [])
        output.mkdir(parents=True, exist_ok=True)
        (output / "missing_data_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        raise MissingDataError(report)

    table_ids = list(TABLE_SPECS) if requested_tables is None else requested_tables
    figure_ids = list(FIGURE_SPECS) if requested_figures is None else requested_figures
    missing_tables: dict[str, list[str]] = {}
    missing_figures: dict[str, list[str]] = {}
    engineering_rows = _validate_no_engineering_or_placeholder(records, allow_engineering_data=allow_engineering_data)
    if engineering_rows:
        report = _missing_report({}, {}, engineering_rows)
        output.mkdir(parents=True, exist_ok=True)
        (output / "missing_data_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        raise MissingDataError(report)

    def _absent_by_design(spec: TableSpec) -> bool:
        """True when this table reports on data the study never collected."""

        if spec.conditional_on is None:
            return False
        return not any(
            row.get(spec.conditional_on) is not None
            or (
                isinstance(row.get("metadata"), dict)
                and row["metadata"].get(spec.conditional_on) is not None
            )
            for row in records
        )

    absent_tables: dict[str, str] = {}
    table_artifacts: list[TableArtifact] = []
    for table_id in table_ids:
        if table_id not in TABLE_SPECS:
            missing_tables[table_id] = ["unknown table id"]
            continue
        spec = TABLE_SPECS[table_id]
        if _absent_by_design(spec):
            absent_tables[table_id] = (
                f"no record carries metadata.{spec.conditional_on}; this study was not run, "
                "so the table is absent rather than incomplete"
            )
            continue
        required_missing = _validate_required_input_columns(records, spec)
        if required_missing:
            missing_tables[table_id] = [f"missing required input columns: {', '.join(required_missing)}"]
            continue
        try:
            artifact = _build_table(records, table_id, bootstrap_resamples=bootstrap_resamples, seed=seed)
        except MissingDataError as exc:
            missing_tables.update(exc.report.get("tables", {}))
            continue
        failures = _validate_table(artifact)
        if failures:
            missing_tables[table_id] = failures
        else:
            table_artifacts.append(artifact)

    figure_artifacts: list[FigureArtifact] = []
    for figure_id in figure_ids:
        if figure_id not in FIGURE_SPECS:
            missing_figures[figure_id] = ["unknown figure id"]
            continue
        try:
            figure = FIGURE_BUILDERS[figure_id](records)
            figure_artifacts.append(figure)
        except MissingDataError as exc:
            missing_figures.update(exc.report.get("figures", {}))

    if missing_tables or missing_figures:
        report = _missing_report(missing_tables, missing_figures, [])
        output.mkdir(parents=True, exist_ok=True)
        (output / "missing_data_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        raise MissingDataError(report)

    table_dir = output / "tables"
    figure_dir = output / "figures"
    manuscript_dir = output / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    numbers: dict[str, Any] = {"tables": {}, "figures": {}, "absent_by_design": absent_tables}
    provenance: dict[str, Any] = {"tables": {}, "figures": {}}
    for artifact in table_artifacts:
        numbers["tables"][artifact.spec.table_id] = _write_table(artifact, table_dir)
        provenance["tables"][artifact.spec.table_id] = {"cells": _cell_provenance(artifact)}
    for figure in figure_artifacts:
        numbers["figures"][figure.spec.figure_id] = _write_figure(figure, figure_dir)
        provenance["figures"][figure.spec.figure_id] = _provenance(figure.source_records)

    numbers["provenance_file"] = "provenance.json"
    numbers["code_commit"] = _current_commit()
    (manuscript_dir / "numbers.json").write_text(json.dumps(_json_safe(numbers), indent=2, sort_keys=True), encoding="utf-8")
    (manuscript_dir / "provenance.json").write_text(json.dumps(_json_safe(provenance), indent=2, sort_keys=True), encoding="utf-8")
    return {"numbers": numbers, "provenance": provenance}


def _current_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate traceable manuscript tables and figures.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--table", action="append", choices=sorted(TABLE_SPECS), help="Generate one table family. Repeatable.")
    parser.add_argument("--figure", action="append", choices=sorted(FIGURE_SPECS), help="Generate one figure. Repeatable.")
    parser.add_argument("--no-figures", action="store_true", help="Generate tables only.")
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--allow-engineering-data", action="store_true", help="Allow authored/synthetic fixtures for development-only output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        generate_artifacts(
            load_records(args.inputs),
            args.output,
            requested_tables=args.table,
            requested_figures=[] if args.no_figures else args.figure,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            allow_engineering_data=args.allow_engineering_data,
        )
    except MissingDataError as exc:
        print(json.dumps(exc.report, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    print(args.output / "manuscript" / "numbers.json")
    print(args.output / "manuscript" / "provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
