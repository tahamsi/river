"""A multi-seed study has to vary with the seed.

RIVER-L is trained under five seeds and reported with intervals across them. If
every seed reaches the same decision on every example, the interval has zero
width and the five runs are effectively one. This happened when the entailment
scores were assigned from the gold label (see docs/integrity.md).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

def _predictions_dir() -> Path:
    """Prefer the most recent benchmark's predictions, without pinning a path."""

    override = os.environ.get("RIVER_PREDICTIONS_DIR")
    if override:
        return Path(override)
    for candidate in ("artifacts/predictions_nli", "artifacts/predictions_measured", "artifacts/predictions"):
        if list(Path(candidate).glob("river_l_test_seed*.jsonl")):
            return Path(candidate)
    return Path("artifacts/predictions")


PREDICTIONS = _predictions_dir()


def decisions(path: Path) -> list[tuple]:
    return [
        (row.get("predicted_state"), row.get("predicted_hypothesis_id"))
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    ]


@pytest.fixture(scope="module")
def seed_runs() -> dict[str, list[tuple]]:
    paths = sorted(PREDICTIONS.glob("river_l_test_seed*.jsonl"))
    if len(paths) < 2:
        pytest.skip("need at least two seed runs")
    return {path.stem: decisions(path) for path in paths}


def test_seeds_do_not_all_agree(seed_runs):
    names = sorted(seed_runs)
    reference = seed_runs[names[0]]
    identical = [
        name for name in names[1:]
        if len(seed_runs[name]) == len(reference)
        and all(left == right for left, right in zip(reference, seed_runs[name]))
    ]
    assert not identical, (
        f"{len(identical) + 1} of {len(names)} seed runs are decision-identical to "
        f"{names[0]}; an interval taken across these seeds has width zero"
    )
