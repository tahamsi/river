r"""Compute the primary endpoints and intervals from the release predictions.

Reads the scored and MiniCheck prediction files of the human-frozen release and
reports, for the 248-example test split:

- state, active-selection, joint and strict-answer accuracy for every system
  (RIVER-L as mean and sample SD over seeds);
- recency flip rate, both pooled over order-reversal and expired-evidence pairs
  (as in the release tables) and over order-reversal pairs alone;
- order-reversal pair outcomes;
- hierarchical bootstrap intervals: test pairs are resampled with replacement
  and RIVER-L seeds are resampled within each draw, and paired differences
  reuse the same sampled pairs;
- the pooled RIVER-L state confusion matrix and per-state recall;
- the sensitivity of RIVER-L joint accuracy to excluding test pairs whose human
  gold contradicts the source-native 2WikiMultihopQA answer.

Usage, after extracting the release archive (see docs/reproducing.md):

    uv run python scripts/release/summarize_release_endpoints.py \
        --release-dir /tmp/river_release/artifacts/release_human_frozen_* \
        --output artifacts/release_endpoints.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from river.evaluation.metrics import recency_flip_rate

STATES = ("active", "abstain", "disputed", "evolution", "unknown")
SYSTEMS = (
    "naive_rag",
    "stateless_rag",
    "last_write_wins",
    "recency_weighted",
    "minicheck_no_revision",
    "river_d",
    "river_l",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def emits_state(rows: list[dict]) -> bool:
    return any(row.get("predicted_state") for row in rows)


def selects_hypotheses(rows: list[dict]) -> bool:
    return any(row.get("predicted_hypothesis_id") for row in rows)


def state_correct(row: dict) -> bool:
    return row.get("predicted_state") == row["gold_state"]


def selection_correct(row: dict) -> bool:
    return row.get("predicted_hypothesis_id") == row["gold_hypothesis_id"]


def joint_correct(row: dict) -> bool:
    if not state_correct(row):
        return False
    return row["gold_hypothesis_id"] is None or selection_correct(row)


def endpoints(rows: list[dict]) -> dict[str, float | None]:
    active = [row for row in rows if row["gold_state"] == "active"]
    strict = [row["answer_text_correct"] for row in rows if row.get("answer_text_correct") is not None]
    structured = emits_state(rows)
    return {
        "state_accuracy": mean(map(state_correct, rows)) if structured else None,
        "active_selection_accuracy": mean(map(selection_correct, active)) if selects_hypotheses(rows) else None,
        "joint_accuracy": mean(map(joint_correct, rows)) if selects_hypotheses(rows) else None,
        "strict_answer_accuracy": mean(map(bool, strict)) if strict else None,
        "rfr_pooled": _finite(recency_flip_rate(rows)),
        "rfr_order_reversal": _finite(
            recency_flip_rate([row for row in rows if row["intervention"] == "order_reversal"])
        ),
    }


def _finite(value: float) -> float | None:
    return None if value != value else value


def order_reversal_outcomes(rows: list[dict]) -> dict[str, float]:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["intervention"] == "order_reversal":
            pairs[row["pair_id"]].append(row)
    counts = Counter(sum(map(joint_correct, members)) for members in pairs.values())
    total = len(pairs)
    return {
        "pairs": total,
        "both_correct": counts[2] / total,
        "one_correct": counts[1] / total,
        "both_wrong": counts[0] / total,
    }


def load_runs(release_dir: Path) -> dict[str, dict[str, list[dict]]]:
    runs: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for folder in ("predictions_scored", "predictions_minicheck"):
        for path in sorted((release_dir / folder).glob("*_test_seed*.jsonl")):
            rows = read_jsonl(path)
            runs[rows[0]["baseline"]][str(rows[0]["seed"])] = rows
    return runs


class PairBootstrap:
    """Resample test pairs, and RIVER-L seeds within each draw."""

    def __init__(self, runs: dict[str, dict[str, list[dict]]], resamples: int, seed: int):
        self.runs = runs
        self.resamples = resamples
        self._cache: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
        self.rng = np.random.default_rng(seed)
        reference = sorted(next(iter(runs["river_l"].values())), key=lambda row: row["example_id"])
        self.example_ids = [row["example_id"] for row in reference]
        for system, by_seed in runs.items():
            for seed, rows in by_seed.items():
                if sorted(row["example_id"] for row in rows) != self.example_ids:
                    raise SystemExit(f"{system} seed {seed} does not cover the same test examples as river_l")
        self.pair_ids = sorted({row["pair_id"] for row in reference})
        index = {pair_id: i for i, pair_id in enumerate(self.pair_ids)}
        # Rows are aligned by example_id everywhere: here and in _vectors.
        self.row_pair = np.array([index[row["pair_id"]] for row in reference])
        self.draws = [
            (
                self.rng.integers(0, len(self.pair_ids), len(self.pair_ids)),
                self.rng.integers(0, len(runs["river_l"]), len(runs["river_l"])),
            )
            for _ in range(resamples)
        ]

    def _weights(self, pair_draw: np.ndarray) -> np.ndarray:
        counts = np.bincount(pair_draw, minlength=len(self.pair_ids))
        return counts[self.row_pair]

    def _vectors(self, system: str, metric) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per run: (metric value per row, 1 if the row is in the metric's denominator)."""

        key = (system, metric.__name__)
        if key in self._cache:
            return self._cache[key]
        vectors = []
        for rows in self.runs[system].values():
            ordered = sorted(rows, key=lambda row: row["example_id"])
            values, mask = zip(*(metric(row) for row in ordered))
            vectors.append((np.array(values, dtype=float), np.array(mask, dtype=float)))
        self._cache[key] = vectors
        return vectors

    def estimate(self, system: str, metric, pair_draw=None, seed_draw=None) -> float:
        vectors = self._vectors(system, metric)
        weights = np.ones(len(self.row_pair)) if pair_draw is None else self._weights(pair_draw)
        chosen = range(len(vectors)) if seed_draw is None or len(vectors) == 1 else seed_draw
        scores = []
        for i in chosen:
            values, mask = vectors[i]
            scores.append(float((values * mask * weights).sum() / (mask * weights).sum()))
        return float(np.mean(scores))

    def interval(self, system: str, metric, minus: str | None = None) -> dict[str, float]:
        def value(pair_draw=None, seed_draw=None):
            result = self.estimate(system, metric, pair_draw, seed_draw)
            if minus is not None:
                result -= self.estimate(minus, metric, pair_draw, seed_draw)
            return result

        samples = [value(pair_draw, seed_draw) for pair_draw, seed_draw in self.draws]
        low, high = np.percentile(samples, [2.5, 97.5])
        return {"estimate": value(), "low": float(low), "high": float(high)}


def metric_state(row):
    return state_correct(row), True


def metric_selection(row):
    return selection_correct(row), row["gold_state"] == "active"


def metric_joint(row):
    return joint_correct(row), True


def metric_strict(row):
    return bool(row.get("answer_text_correct")), row.get("answer_text_correct") is not None


def confusion(rows_by_seed: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    matrix = {gold: {predicted: 0 for predicted in STATES} for gold in STATES}
    for rows in rows_by_seed.values():
        for row in rows:
            matrix[row["gold_state"]][row["predicted_state"]] += 1
    return matrix


def source_answer_conflicts(fixture: Path) -> set[str]:
    """Test pairs whose active gold contradicts the source-native 2WikiMultihopQA answer."""

    conflicts = set()
    for row in read_jsonl(fixture):
        record = row["transformation_metadata"].get("source_native_record") or {}
        if row["source_dataset"] != "2wikimultihopqa" or row["gold_state"] != "active" or "answer" not in record:
            continue
        gold = next(h["object"] for h in row["hypotheses"] if h["hypothesis_id"] == row["gold_hypothesis_id"])
        if gold.strip().lower() != str(record["answer"]).strip().lower():
            conflicts.add(row["pair_id"])
    return conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("data/grounded_human_frozen/river_bench_test.jsonl"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = load_runs(args.release_dir)
    missing = [system for system in SYSTEMS if system not in runs]
    if missing:
        raise SystemExit(f"missing prediction files for: {', '.join(missing)}")

    systems = {}
    for system in SYSTEMS:
        per_seed = {seed: endpoints(rows) for seed, rows in runs[system].items()}
        summary = {}
        for key in next(iter(per_seed.values())):
            values = [result[key] for result in per_seed.values() if result[key] is not None]
            summary[key] = {
                "mean": mean(values) if values else None,
                "sample_sd": stdev(values) if len(values) > 1 else None,
            }
        systems[system] = {"seeds": sorted(per_seed, key=int), "endpoints": summary}
        if system in {"last_write_wins", "river_d", "river_l"}:
            outcomes = [order_reversal_outcomes(rows) for rows in runs[system].values()]
            systems[system]["order_reversal_pairs"] = {
                key: mean(result[key] for result in outcomes) for key in outcomes[0]
            }

    bootstrap = PairBootstrap(runs, args.resamples, args.seed)
    intervals = {
        "river_l_state_accuracy": bootstrap.interval("river_l", metric_state),
        "river_l_active_selection": bootstrap.interval("river_l", metric_selection),
        "river_l_joint_accuracy": bootstrap.interval("river_l", metric_joint),
        "river_l_strict_answer": bootstrap.interval("river_l", metric_strict),
        "last_write_wins_joint_accuracy": bootstrap.interval("last_write_wins", metric_joint),
        "river_d_joint_accuracy": bootstrap.interval("river_d", metric_joint),
        "river_l_minus_last_write_wins_joint": bootstrap.interval("river_l", metric_joint, minus="last_write_wins"),
        "river_l_minus_naive_rag_strict_answer": bootstrap.interval("river_l", metric_strict, minus="naive_rag"),
    }

    matrix = confusion(runs["river_l"])
    recall = {
        state: (matrix[state][state] / sum(matrix[state].values())) if sum(matrix[state].values()) else None
        for state in STATES
    }

    conflicts = source_answer_conflicts(args.fixture)
    sensitivity = {
        "excluded_pairs": sorted(conflicts),
        "river_l_joint_accuracy_excluding": mean(
            mean(joint_correct(row) for row in rows if row["pair_id"] not in conflicts)
            for rows in runs["river_l"].values()
        ),
    }

    report = {
        "release_dir": str(args.release_dir),
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "unit": "test pair; RIVER-L seeds resampled within each draw"},
        "systems": systems,
        "intervals": intervals,
        "river_l_confusion_pooled_over_seeds": matrix,
        "river_l_state_recall": recall,
        "source_answer_sensitivity": sensitivity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'system':24s} {'state':>7s} {'select':>7s} {'joint':>7s} {'strict':>7s} {'RFR':>7s} {'RFR-OR':>7s}")
    for system, result in systems.items():
        cells = [result["endpoints"][key]["mean"] for key in (
            "state_accuracy", "active_selection_accuracy", "joint_accuracy",
            "strict_answer_accuracy", "rfr_pooled", "rfr_order_reversal",
        )]
        print(f"{system:24s} " + " ".join("      –" if c is None else f"{c:7.4f}" for c in cells))
    print()
    for name, interval in intervals.items():
        print(f"{name:40s} {interval['estimate']:.4f} [{interval['low']:.4f}, {interval['high']:.4f}]")
    print(f"\nRIVER-L joint excluding {len(conflicts)} source-conflict test pair(s): "
          f"{sensitivity['river_l_joint_accuracy_excluding']:.4f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
