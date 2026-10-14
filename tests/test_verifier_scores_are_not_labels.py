"""The stored NLI scores must not identify the gold answer on their own.

Each evidence item ships entailment, contradiction and neutral values. A system
reads them as features; they are RIVER-L's first three inputs. If those numbers
are assigned from the construction rather than measured from the text, then
picking the hypothesis with the highest entailment answers the benchmark without
reading a word of evidence, and every metric built on hypothesis selection
measures the assignment instead of the reasoning.

Chance on a two-hypothesis row is 50%. A score well above that means the label
is recoverable from a field no reader of the evidence would consult.

These pass against corpora whose scores were measured by a verifier. They fail
against data/grounded/, where the scores were assigned, which is why that
directory is superseded for reporting.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

GROUNDED = Path(os.environ.get("RIVER_GROUNDED_DIR", "data/grounded_nli"))
SPLITS = ("train", "dev", "test")
# Two hypotheses is the common case, so an unbiased field lands near 0.5. A real
# verifier lands well above that, because it is reading evidence that does
# support one side: the highest observed on measured scores is 0.875, on
# unsupported_late_assertion, whose construction gives the gold hypothesis
# one-sided support by design.
#
# The bound is therefore set to catch assignment rather than skill. Scores
# written from the gold label send every family to 1.000 at once, which this
# catches, and test_the_scores_are_not_drawn_from_two_fixed_values catches the
# same fault independently by counting distinct values.
UPPER_BOUND = 0.95


def load() -> list[dict]:
    paths = [GROUNDED / f"river_bench_{split}.jsonl" for split in SPLITS]
    if not all(path.exists() for path in paths):
        pytest.skip(f"benchmark splits not present under {GROUNDED}")
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def contested() -> list[dict]:
    """Rows where a gold answer competes with at least one alternative."""

    selected = []
    for row in load():
        if row.get("gold_hypothesis_id") is None:
            continue
        by_hypothesis: dict[str, float] = {}
        for evidence in row.get("evidence") or []:
            hypothesis_id = evidence.get("hypothesis_id")
            if hypothesis_id is None:
                continue
            score = float(evidence.get("entailment") or 0.0)
            by_hypothesis[hypothesis_id] = max(by_hypothesis.get(hypothesis_id, -1.0), score)
        if len(by_hypothesis) >= 2:
            selected.append({"row": row, "scores": by_hypothesis})
    if not selected:
        pytest.skip("no rows with competing hypotheses")
    return selected


def test_entailment_does_not_identify_the_gold_answer(contested):
    by_family: dict[str, list[bool]] = defaultdict(list)
    for entry in contested:
        row, scores = entry["row"], entry["scores"]
        picked = max(scores, key=lambda key: scores[key])
        by_family[row["intervention"]].append(picked == row["gold_hypothesis_id"])

    offenders = {
        family: round(sum(hits) / len(hits), 3)
        for family, hits in by_family.items()
        if sum(hits) / len(hits) > UPPER_BOUND
    }
    assert not offenders, (
        "gold is recoverable from the stored entailment scores alone: "
        f"{offenders}. These values must be measured against the evidence text "
        "by a verifier, not assigned from the gold label at construction time."
    )


def test_the_scores_are_not_drawn_from_two_fixed_values(contested):
    """A measured score varies; an assigned one repeats.

    The construction wrote one constant for supporting evidence and another for
    distractors, so the whole benchmark used a handful of distinct numbers. Real
    verifier output over thousands of sentence pairs does not do that.
    """

    values = {
        round(float(evidence.get("entailment") or 0.0), 4)
        for entry in contested
        for evidence in entry["row"].get("evidence") or []
    }
    assert len(values) > 20, (
        f"only {len(values)} distinct entailment values across the benchmark "
        f"({sorted(values)}); these look assigned rather than measured"
    )
