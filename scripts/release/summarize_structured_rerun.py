#!/usr/bin/env python3
"""Summarize structured test decisions without treating IDs as answer text."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def wilson(hits: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not total:
        return [float("nan"), float("nan")]
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [centre - radius, centre + radius]


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    applicable = any(row.get("predicted_state") is not None for row in rows)
    if not applicable:
        return {
            "baseline": rows[0]["baseline"],
            "seed": rows[0]["seed"],
            "examples": len(rows),
            "structured_metrics_applicable": False,
            "reason": "baseline does not emit a structured decision state",
        }
    state_hits = sum(str(row["predicted_state"]).lower() == str(row["gold_state"]).lower() for row in rows)
    active = [row for row in rows if row.get("gold_hypothesis_id") is not None]
    selection_hits = sum(row.get("predicted_hypothesis_id") == row["gold_hypothesis_id"] for row in active)
    joint_hits = sum(bool(row.get("answer_correct")) for row in rows)
    interventions = defaultdict(list)
    for row in rows:
        interventions[row["intervention"]].append(bool(row.get("answer_correct")))
    return {
        "baseline": rows[0]["baseline"],
        "seed": rows[0]["seed"],
        "examples": len(rows),
        "structured_metrics_applicable": True,
        "state_accuracy": state_hits / len(rows),
        "state_accuracy_wilson95": wilson(state_hits, len(rows)),
        "active_selection_accuracy": selection_hits / len(active),
        "active_selection_accuracy_wilson95": wilson(selection_hits, len(active)),
        "active_selection_n": len(active),
        "joint_structured_accuracy": joint_hits / len(rows),
        "joint_structured_accuracy_wilson95": wilson(joint_hits, len(rows)),
        "by_intervention_joint_accuracy": {
            key: {"n": len(values), "accuracy": sum(values) / len(values)}
            for key, values in sorted(interventions.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [summarize(path) for path in sorted(args.predictions.glob("*_test_seed*.jsonl"))]
    river_l = [row for row in runs if row["baseline"] == "river_l"]
    aggregate = {}
    if river_l:
        for metric in ("state_accuracy", "active_selection_accuracy", "joint_structured_accuracy"):
            values = [row[metric] for row in river_l]
            aggregate[metric] = {"mean": mean(values), "sample_sd": stdev(values)}
    result = {
        "scope": "structured decisions only; no generated-answer or citation claims",
        "runs": runs,
        "river_l_five_seed_aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
