"""Fit RIVER-D's decision thresholds on the dev split.

The default thresholds in configs/baselines/river_d.json (support 0.6, margin
0.15, abstention 0.3) were set for scores that took only two values, 1.0 and
0.45. Measured NLI scores are distributed differently, so the thresholds are
refitted.

Selection uses the dev split only; the test split is read once, after the
thresholds are fixed. The objective is the mean of state accuracy and
hypothesis-selection accuracy, because state accuracy alone rewards a
configuration that abstains instead of choosing between competing claims.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from river import build_baseline

ROOT = Path(__file__).resolve().parents[2]


def evidence_for_baseline(item: dict) -> list[dict]:
    rows = []
    for evidence in item.get("evidence", []):
        if not isinstance(evidence, dict):
            rows.append(evidence)
            continue
        normalized = dict(evidence)
        if "transaction_time" in normalized and "observed_at" not in normalized:
            normalized["observed_at"] = normalized["transaction_time"]
        rows.append(normalized)
    return rows


def evaluate(rows: list[dict], config: dict) -> tuple[float, float]:
    baseline = build_baseline("river_d", dict(config))
    state_hits = 0
    selection_hits = 0
    contested = 0
    for item in rows:
        if item.get("hypotheses") is not None and hasattr(baseline, "config"):
            baseline.config["hypotheses"] = item["hypotheses"]
        result = baseline.process_turn(
            item.get("session_id", item["base_example_id"]),
            item["query"],
            evidence_for_baseline(item),
            item.get("target_time"),
        )
        gold_state = str(item.get("gold_state") or "").lower()
        state_hits += str(result.get("predicted_state") or "").lower() == gold_state
        gold_hypothesis = item.get("gold_hypothesis_id")
        if gold_hypothesis is not None:
            contested += 1
            selection_hits += result.get("predicted_hypothesis_id") == gold_hypothesis
    return state_hits / len(rows), (selection_hits / contested if contested else float("nan"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=ROOT / "data/grounded_human_frozen/river_bench_dev.jsonl")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/baselines/river_d.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/river_d_tuned.json")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dev.read_text(encoding="utf-8").splitlines() if line.strip()]
    base = json.loads(args.base_config.read_text(encoding="utf-8"))

    supports = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    margins = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25]
    abstentions = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    shipped_state, shipped_selection = evaluate(rows, base)
    print(f"shipped thresholds on dev: state {shipped_state:.4f}  selection {shipped_selection:.4f}\n")

    best = None
    for support, margin, abstention in product(supports, margins, abstentions):
        config = dict(base)
        config.update(
            support_threshold=support,
            margin_threshold=margin,
            abstention_threshold=abstention,
        )
        state, selection = evaluate(rows, config)
        score = (state + selection) / 2
        if best is None or score > best[0]:
            best = (score, support, margin, abstention, state, selection)
            print(f"  support {support:<4} margin {margin:<5} abstain {abstention:<4} -> state {state:.4f} selection {selection:.4f}")

    score, support, margin, abstention, state, selection = best
    print(f"\nbest on dev: support {support}, margin {margin}, abstention {abstention}")
    print(f"  state {state:.4f}  selection {selection:.4f}  mean {score:.4f}")

    tuned = dict(base)
    tuned.update(
        support_threshold=support,
        margin_threshold=margin,
        abstention_threshold=abstention,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tuned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
