"""Families reported as case studies stay out of headline aggregates.

retroactive_correction contains evidence text that could not be anchored in the
cited source record, and has too few pairs to support an interval. It is
reported as a counted case study. supported_genuine_update and
temporal_evolution were reviewed row by row and approved on 2026-09-18, and are
included in the aggregates (see docs/integrity.md). This test fails if
retroactive_correction is added back to a headline aggregate.
"""

from __future__ import annotations

import pytest

from analysis.generate_tables import (
    TABLE_METRICS,
    _controlled,
    _controlled_including_qualitative,
)
from river.evaluation.metrics import QUALITATIVE_ONLY_INTERVENTIONS


HEADLINE_TABLES = (
    "overall_controlled_baseline_comparison",
    "paired_recency_decomposition",
    "main_river_ablation",
    "river_d_vs_river_l",
    "conflict_detection_negative_controls",
    "nei_selective_abstention",
)

CASE_STUDY_TABLES = ("results_by_intervention",)


def row(intervention: str) -> dict[str, object]:
    return {"comparison_mode": "controlled", "intervention": intervention, "baseline": "river_l"}


def test_the_registered_set_is_the_audited_one():
    assert QUALITATIVE_ONLY_INTERVENTIONS == {"retroactive_correction"}


@pytest.mark.parametrize("intervention", sorted(QUALITATIVE_ONLY_INTERVENTIONS))
def test_qualitative_families_are_not_controlled(intervention):
    assert _controlled(row(intervention)) is False
    assert _controlled_including_qualitative(row(intervention)) is True


@pytest.mark.parametrize("intervention", ["order_reversal", "concurrent_disagreement"])
def test_quantitative_families_still_pass(intervention):
    assert _controlled(row(intervention)) is True


@pytest.mark.parametrize("table_id", HEADLINE_TABLES)
def test_headline_tables_admit_no_qualitative_row(table_id):
    """Every defended table must reject all three families, whatever its own filter."""

    _columns, predicate, _group = TABLE_METRICS[table_id]
    for intervention in QUALITATIVE_ONLY_INTERVENTIONS:
        candidate = row(intervention)
        candidate["metadata"] = {"condition": "river_d_full", "mechanism": "deterministic_revision"}
        candidate["citation_supported"] = True
        candidate["answer_correct"] = True
        assert predicate(candidate) is False, f"{table_id} admitted {intervention}"


@pytest.mark.parametrize("table_id", CASE_STUDY_TABLES)
def test_case_study_tables_keep_the_families(table_id):
    _columns, predicate, group = TABLE_METRICS[table_id]
    assert "intervention" in group, "a case-study table must group by family"
    for intervention in QUALITATIVE_ONLY_INTERVENTIONS:
        assert predicate(row(intervention)) is True, f"{table_id} dropped {intervention}"
