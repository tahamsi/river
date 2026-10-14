"""How much of a result depends on the stored verifier scores.

Runs the revision operator twice over the same split, once as given and once
with the entailment, contradiction and neutral fields flattened so they carry no
information, and reports the difference. Generation is disabled, so this costs
no inference.

The flattened run is a lower bound rather than an estimate of the system
without the scores, because it also removes the signal a real verifier
legitimately supplies. It was used to show how much an early corpus, whose
scores were assigned from the gold label, inflated results (see
docs/integrity.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEUTRAL_SCORES = {"entailment": 0.5, "contradiction": 0.0, "neutral": 0.5}


def flatten(fixture: Path, destination: Path) -> None:
    """Copy the split with the verifier fields carrying no information."""

    with destination.open("w", encoding="utf-8") as handle:
        for line in fixture.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for evidence in row.get("evidence") or []:
                evidence.update(NEUTRAL_SCORES)
            handle.write(json.dumps(row) + "\n")


def run(fixture: Path, config: Path, output: Path, baseline: str) -> None:
    result = subprocess.run(
        [
            sys.executable, "-m", "experiments.run",
            "--baseline", baseline, "--fixture", str(fixture),
            "--config", str(config), "--seed", "42",
            "--no-generation", "--output", str(output),
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"run failed for {fixture.name}:\n{result.stderr[-2000:]}")


def accuracy(predictions: Path, gold: dict[str, tuple]) -> tuple[float, float, int]:
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    contested = [row for row in rows if gold[row["example_id"]][1] is not None]
    selection = sum(
        1 for row in contested if row.get("predicted_hypothesis_id") == gold[row["example_id"]][1]
    )
    state = sum(
        1 for row in rows
        if str(row.get("predicted_state") or "").lower() == str(gold[row["example_id"]][0] or "").lower()
    )
    return selection / len(contested), state / len(rows), len(contested)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "data/grounded_human_frozen/river_bench_test.jsonl")
    parser.add_argument("--baseline", default="river_d")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baselines/river_d.json")
    args = parser.parse_args()

    gold = {
        row["example_id"]: (row.get("gold_state"), row.get("gold_hypothesis_id"))
        for row in map(json.loads, args.fixture.read_text(encoding="utf-8").splitlines())
    }

    with tempfile.TemporaryDirectory() as workspace:
        work = Path(workspace)
        flattened = work / "flattened.jsonl"
        flatten(args.fixture, flattened)

        as_released = work / "released.jsonl"
        without = work / "flattened_predictions.jsonl"
        run(args.fixture, args.config, as_released, args.baseline)
        run(flattened, args.config, without, args.baseline)

        released_selection, released_state, contested = accuracy(as_released, gold)
        flat_selection, flat_state, _ = accuracy(without, gold)

    print(f"{args.baseline} on {args.fixture.name}, {contested} contested rows\n")
    print(f"{'condition':22} {'selection':>10} {'state':>10}")
    print(f"{'as released':22} {released_selection:10.3f} {released_state:10.3f}")
    print(f"{'scores flattened':22} {flat_selection:10.3f} {flat_state:10.3f}")
    print(f"{'attributable to them':22} {released_selection - flat_selection:10.3f} {released_state - flat_state:10.3f}")
    print(
        "\nThe flattened row is a floor: it removes every verification signal, "
        "not only the leaked one.\nThe system's true accuracy lies between these "
        "numbers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
