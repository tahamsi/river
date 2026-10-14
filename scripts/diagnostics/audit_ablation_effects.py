"""Check that each ablated mechanism actually changes a decision.

An ablation table reports what happens when a mechanism is switched off. If the
switch changes nothing, the row is not evidence that the mechanism is
unnecessary; it is evidence that the data never exercised it. The two read
identically in a table and mean opposite things, so the difference has to be
measured rather than assumed.

Runs each ablation with generation disabled, so it costs no inference: the
revision operator is deterministic and its decisions are what an ablation
changes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(config: Path, fixture: Path, output: Path, baseline: str) -> None:
    command = [
        sys.executable, "-m", "experiments.run",
        "--baseline", baseline,
        "--fixture", str(fixture),
        "--config", str(config),
        "--seed", "42",
        "--no-generation",
        "--output", str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{config.name} failed:\n{result.stderr[-2000:]}")


def decisions(path: Path) -> list[tuple]:
    return [
        (row.get("predicted_state"), row.get("predicted_hypothesis_id"))
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "data/grounded/river_bench_test.jsonl")
    parser.add_argument("--baseline", default="river_d")
    parser.add_argument("--full-config", type=Path, default=ROOT / "configs/baselines/river_d.json")
    parser.add_argument("--ablation-dir", type=Path, default=ROOT / "configs/ablations")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workspace:
        work = Path(workspace)
        baseline_path = work / "full.jsonl"
        run(args.full_config, args.fixture, baseline_path, args.baseline)
        reference = decisions(baseline_path)

        rows = []
        for config in sorted(args.ablation_dir.glob("*.json")):
            output = work / f"{config.stem}.jsonl"
            run(config, args.fixture, output, args.baseline)
            changed = sum(1 for left, right in zip(reference, decisions(output)) if left != right)
            rows.append((config.stem, changed))

    total = len(reference)
    width = max(len(name) for name, _ in rows)
    inert = [name for name, changed in rows if changed == 0]
    print(f"{'ablation':{width}}  changed  of {total}")
    for name, changed in rows:
        note = "  INERT" if changed == 0 else ""
        print(f"{name:{width}}  {changed:7d}  {total}{note}")
    print()
    if inert:
        print(f"{len(inert)} of {len(rows)} ablated mechanisms changed no decision:")
        for name in inert:
            print(f"  - {name}")
        print(
            "An ablation row for one of these reports the data, not the mechanism.\n"
            "Reporting it as evidence the mechanism does not matter would be a\n"
            "claim these runs cannot support."
        )
        return 1
    print("every ablated mechanism changed at least one decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
