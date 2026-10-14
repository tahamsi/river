"""Evidence features that a system reads must carry information.

RIVER-L reads entailment, contradiction, extraction confidence and provenance
reliability among its fourteen features, and RIVER-D's ablations switch off
mechanisms that use them. A field with a single value across the benchmark
cannot influence a decision, so ablating it changes nothing by construction.

Entailment and contradiction are measured by the NLI model and vary.
Provenance reliability and extraction confidence are constant (1.0) in this
release and are marked as expected failures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

GROUNDED = Path(os.environ.get("RIVER_GROUNDED_DIR", "data/grounded_nli"))
SPLITS = ("train", "dev", "test")
FIELDS = ("extraction_confidence", "provenance_reliability")


@pytest.fixture(scope="module")
def evidence() -> list[dict]:
    paths = [GROUNDED / f"river_bench_{split}.jsonl" for split in SPLITS]
    if not all(path.exists() for path in paths):
        pytest.skip(f"benchmark splits not present under {GROUNDED}")
    items = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.extend(json.loads(line).get("evidence") or [])
    return items


@pytest.mark.xfail(
    strict=True,
    reason="never measured. These are provenance metadata the build set to 1.0 "
    "everywhere, not verifier output, so the NLI pass does not touch them. Any "
    "mechanism reading them acts on a constant, which is why use_provenance "
    "cannot be tested; see docs/integrity.md",
)
@pytest.mark.parametrize("field", FIELDS)
def test_feature_is_not_constant(evidence, field):
    values = {round(float(item.get(field) or 0.0), 6) for item in evidence}
    assert len(values) > 1, (
        f"{field} holds the single value {values.pop()} across "
        f"{len(evidence)} evidence items, so it cannot influence any decision"
    )
