"""The benchmark must contain evidence that is not valid at the target time.

RIVER uses evidence when its proposition holds at the time asked about, not when
the evidence arrived. Testing that requires cases where an interval excludes the
target time, so that honouring valid time and ignoring it give different
decisions. Without them, `river_no_valid_time` changes nothing and an ablation
table would wrongly suggest that valid time does not matter.

The released corpus provides these cases through the expired_evidence family.
The earlier corpus in data/grounded_nli/ did not, which is recorded as an
expected failure.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

GROUNDED = Path(os.environ.get("RIVER_GROUNDED_DIR", "data/grounded_nli"))
SPLITS = ("train", "dev", "test")


def parse(value) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


@pytest.fixture(scope="module")
def applicability() -> tuple[int, int]:
    """(inside, outside) counts over evidence that could be judged at all."""

    paths = [GROUNDED / f"river_bench_{split}.jsonl" for split in SPLITS]
    if not all(path.exists() for path in paths):
        pytest.skip(f"benchmark splits not present under {GROUNDED}")

    inside = outside = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            target = parse(row.get("target_time"))
            if target is None:
                continue
            for evidence in row.get("evidence") or []:
                interval = evidence.get("valid_time") or {}
                start, end = parse(interval.get("start")), parse(interval.get("end"))
                if start is None and end is None:
                    continue
                if (start is None or target >= start) and (end is None or target <= end):
                    inside += 1
                else:
                    outside += 1
    if inside + outside == 0:
        pytest.skip("no evidence carries both a target time and a validity interval")
    return inside, outside


@pytest.mark.xfail(
    strict=True,
    reason="no evidence in data/grounded_nli/ is invalid at the time asked about, "
    "so use_valid_time cannot change a decision there. data/grounded_expired_nli/ "
    "is a candidate family that does exercise it and passes this test; the marker "
    "comes off when that family is merged into the reported benchmark. See "
    "docs/integrity.md",
)
def test_some_evidence_is_invalid_at_the_target_time(applicability):
    inside, outside = applicability
    assert outside > 0, (
        f"all {inside} judgeable evidence items are valid at their target time. "
        "Honouring valid time and ignoring it cannot differ on this data, so the "
        "mechanism is untested rather than unimportant."
    )


RELEASED = Path(os.environ.get("RIVER_RELEASED_DIR", "data/grounded_human_frozen"))


def _applicability(directory: Path) -> tuple[int, int]:
    paths = [directory / f"river_bench_{split}.jsonl" for split in SPLITS]
    if not all(path.exists() for path in paths):
        pytest.skip(f"benchmark splits not present under {directory}")

    inside = outside = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            target = parse(row.get("target_time"))
            if target is None:
                continue
            for evidence in row.get("evidence") or []:
                interval = evidence.get("valid_time") or {}
                start, end = parse(interval.get("start")), parse(interval.get("end"))
                if start is None and end is None:
                    continue
                if (start is None or target >= start) and (end is None or target <= end):
                    inside += 1
                else:
                    outside += 1
    return inside, outside


def test_released_corpus_exercises_valid_time():
    """The released corpus must contain evidence that is invalid at its target time.

    The expired_evidence family supplies it. Without such rows, switching
    use_valid_time off could not change a decision, and the temporal ablation
    would be vacuous.
    """

    inside, outside = _applicability(RELEASED)
    assert outside > 0, (
        f"all {inside} judgeable evidence items in {RELEASED} are valid at their "
        "target time, so the temporal ablation is vacuous"
    )
