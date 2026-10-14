"""Baselines reported as separate rows must behave differently.

On this benchmark naive RAG and stateless RAG produce identical predictions, as
do last-write-wins and recency-weighted. Reporting such a pair as two systems
overstates how many independent comparisons a table contains. This test reads
the available prediction files and fails when two baselines are
indistinguishable, unless the pair is declared in ``KNOWN_EQUIVALENT`` and
reported as identical (see docs/integrity.md).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

PREDICTIONS = Path(os.environ.get("RIVER_PREDICTIONS_DIR", "artifacts/predictions"))
COMPARED_FIELDS = ("predicted_state", "predicted_hypothesis_id", "answer")

# Equivalences already established, with the reason each holds. A pair listed
# here is still reported as one system, not two; the entry records that the
# duplication is understood rather than unnoticed.
KNOWN_EQUIVALENT = {
    frozenset({"naive_rag", "stateless_rag"}):
        "StatelessRAG subclasses NaiveRAG and overrides only its name.",
    frozenset({"last_write_wins", "recency_weighted"}):
        "RIVER-Bench supplies two evidence items per row and no retrieval_score, "
        "so the recency-weighted score reduces to picking the newer item.",
}


def load(path: Path) -> dict[str, tuple]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["example_id"]] = tuple(row.get(field) for field in COMPARED_FIELDS)
    return rows


@pytest.fixture(scope="module")
def by_baseline() -> dict[str, dict[str, tuple]]:
    if not PREDICTIONS.is_dir():
        pytest.skip(f"no prediction directory at {PREDICTIONS}")
    grouped: dict[str, dict[str, tuple]] = defaultdict(dict)
    for path in sorted(PREDICTIONS.glob("*_test_seed42.jsonl")):
        name = path.name.removesuffix("_test_seed42.jsonl")
        grouped[name] = load(path)
    if len(grouped) < 2:
        pytest.skip("fewer than two test-split prediction files")
    return grouped


def test_no_unacknowledged_duplicate_baselines(by_baseline):
    names = sorted(by_baseline)
    duplicates = []
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = by_baseline[left].keys() & by_baseline[right].keys()
            if not shared:
                continue
            if all(by_baseline[left][key] == by_baseline[right][key] for key in shared):
                if frozenset({left, right}) not in KNOWN_EQUIVALENT:
                    duplicates.append(f"{left} == {right} on {len(shared)} rows")
    assert not duplicates, "baselines are indistinguishable: " + "; ".join(duplicates)


@pytest.mark.parametrize("pair", sorted(KNOWN_EQUIVALENT, key=sorted))
def test_known_equivalences_still_hold(by_baseline, pair):
    """If one of these ever diverges, it should be reported as two systems again."""

    left, right = sorted(pair)
    if left not in by_baseline or right not in by_baseline:
        pytest.skip(f"predictions missing for {left} or {right}")
    shared = by_baseline[left].keys() & by_baseline[right].keys()
    assert shared, f"no shared examples between {left} and {right}"
    identical = all(by_baseline[left][key] == by_baseline[right][key] for key in shared)
    assert identical, (
        f"{left} and {right} now differ; they are recorded as equivalent in "
        "KNOWN_EQUIVALENT and should be reported separately if that is real"
    )
