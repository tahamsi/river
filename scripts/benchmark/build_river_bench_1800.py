"""Build the deterministic 1,500-row source corpus manifest for RIVER-Bench.

Selects the fixed quotas from the five source datasets and records hashes and
source identifiers. It assigns no RIVER labels and refuses to overwrite an
existing output. Annotation into the BenchmarkExample schema is a separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from river_bench.loaders.external import DATASET_SPECS


SOURCE_QUOTAS = {
    "fever": 350,
    "pat_questions": 300,
    "wikicontradict": 250,
    "longmemeval": 300,
    "2wikimultihopqa": 300,
}
SOURCES = tuple(SOURCE_QUOTAS)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stable_id(dataset_id: str, row: dict[str, Any], source_row_index: int) -> str:
    if dataset_id == "pat_questions" and row.get("uniq_id") is not None:
        task = row.get("_river_source_task", "unknown")
        return f"{dataset_id}:{task}:{row['uniq_id']}:row:{source_row_index}"
    for key in ("id", "_id", "qa_id", "claim_id", "question_id"):
        if row.get(key) is not None:
            return f"{dataset_id}:{row[key]}:row:{source_row_index}"
    digest = hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()
    return f"{dataset_id}:sha256:{digest}:row:{source_row_index}"


def build(cache_dir: Path, output: Path, *, seed: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for dataset_id in SOURCES:
        # Source-specific seeds make each source selection stable when another
        # source is added to or removed from the composite.
        rng = random.Random(f"{seed}:{dataset_id}")
        quota = SOURCE_QUOTAS[dataset_id]
        spec = DATASET_SPECS[dataset_id]
        candidates = [
            cache_dir / dataset_id / f"{spec.default_split or 'default'}.jsonl",
            cache_dir / dataset_id / "manual.jsonl",
        ]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(
                f"missing cached source {dataset_id}; run `river-download --fail-fast` "
                f"or provide {cache_dir / dataset_id / 'manual.jsonl'}"
            )
        rows = _read(source)
        if len(rows) < quota:
            raise ValueError(f"{dataset_id} has {len(rows)} rows; {quota} are required")
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        picked = sorted(indices[:quota])
        counts[dataset_id] = len(picked)
        source_hash = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
        for index in picked:
            row = rows[index]
            selected.append({
                "river_source_id": _stable_id(dataset_id, row, index),
                "source_dataset": dataset_id,
                "source_split": (
                    row.get("_river_source_snapshot")
                    or ("longmemeval_oracle" if dataset_id == "longmemeval" else spec.default_split)
                ),
                "source_row_index": index,
                "source_file_hash": source_hash,
                "raw_record": row,
            })
    if len(selected) != sum(SOURCE_QUOTAS.values()):
        raise AssertionError("unexpected composite size")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in selected), encoding="utf-8")
    manifest = {
        "schema_version": "river-source-corpus-v1",
        "scientific_finalization_eligible": False,
        "reason": "Source selection is complete; RIVER intervention annotation and schema normalization are still required.",
        "seed": seed,
        "source_quotas": SOURCE_QUOTAS,
        "total": len(selected),
        "counts": counts,
        "output_sha256": "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/river_source_corpus_1500.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.cache_dir, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
