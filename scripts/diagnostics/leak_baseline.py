"""A baseline that reads only the stored verifier scores.

It ignores the evidence text, timestamps, valid time and the query. It sorts the
stored entailment values and applies three thresholds. Anything it achieves is
information the benchmark gives away through the scores rather than a test of
revision. On a corpus whose scores were assigned from the gold label it matched
RIVER-D's state accuracy (see docs/integrity.md).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Supporting evidence was assigned values at or above 0.86 and competing
# evidence at or below 0.45, so any threshold between them separates them.
SUPPORT_FLOOR = 0.5
# Two hypotheses assigned the same constant are the concurrent-disagreement
# construction, which is what "disputed" means in this benchmark.
TIE_WIDTH = 0.05


def predict(row: dict) -> str:
    best: dict[str, float] = {}
    for evidence in row.get("evidence") or []:
        hypothesis_id = evidence.get("hypothesis_id")
        if hypothesis_id is None:
            continue
        score = float(evidence.get("entailment") or 0.0)
        best[hypothesis_id] = max(best.get(hypothesis_id, -1.0), score)
    if not best:
        return "abstain"
    ranked = sorted(best.values(), reverse=True)
    if ranked[0] < SUPPORT_FLOOR:
        return "abstain"
    if len(ranked) >= 2 and abs(ranked[0] - ranked[1]) < TIE_WIDTH:
        return "disputed"
    return "active"


def select(row: dict) -> str | None:
    best: dict[str, float] = {}
    for evidence in row.get("evidence") or []:
        hypothesis_id = evidence.get("hypothesis_id")
        if hypothesis_id is None:
            continue
        score = float(evidence.get("entailment") or 0.0)
        best[hypothesis_id] = max(best.get(hypothesis_id, -1.0), score)
    if not best:
        return None
    return max(best, key=lambda key: best[key])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("data/grounded_human_frozen/river_bench_test.jsonl"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.fixture.read_text(encoding="utf-8").splitlines() if line.strip()]

    state_hits = 0
    selection_hits = 0
    contested = 0
    confusion: Counter[tuple[str, str]] = Counter()

    for row in rows:
        gold_state = str(row.get("gold_state") or "").lower()
        predicted = predict(row)
        state_hits += predicted == gold_state
        confusion[(gold_state, predicted)] += 1
        gold_hypothesis = row.get("gold_hypothesis_id")
        if gold_hypothesis is not None:
            contested += 1
            selection_hits += select(row) == gold_hypothesis

    print(f"{args.fixture.name}: {len(rows)} rows, {contested} with a gold hypothesis\n")
    print(f"state accuracy      {state_hits / len(rows):.4f}  ({state_hits}/{len(rows)})")
    print(f"hypothesis selection {selection_hits / contested:.4f}  ({selection_hits}/{contested})")
    print("\nconfusion:")
    for (gold, predicted), count in sorted(confusion.items()):
        marker = "" if gold == predicted else "   <- wrong"
        print(f"  gold {gold:9} -> {predicted:9} {count:4d}{marker}")
    print(
        "\nThis reads no evidence text, no timestamps and no valid time.\n"
        "Whatever it scores is what the stored fields give away."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
