"""Download/cache every dataset declared by the frozen RIVER experiment plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from river_bench.loaders.external import DATASET_SPECS, download_external_dataset


def download_all(cache_dir: Path, *, limit: int | None, manual_dir: Path | None, fail_fast: bool = False) -> dict:
    results = []
    for dataset_id in sorted(DATASET_SPECS):
        manual_file = None
        if manual_dir:
            for suffix in (".jsonl",):
                candidate = manual_dir / f"{dataset_id}{suffix}"
                if candidate.exists():
                    manual_file = candidate
                    break
        try:
            path = download_external_dataset(
                dataset_id,
                cache_dir=cache_dir,
                limit=limit,
                manual_file=manual_file,
            )
            rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            results.append({"dataset_id": dataset_id, "status": "downloaded", "path": str(path), "rows": rows})
        except (NotImplementedError, PermissionError) as exc:
            results.append({"dataset_id": dataset_id, "status": "manual_action_required", "reason": str(exc)})
            if fail_fast:
                raise
        except Exception as exc:
            results.append({"dataset_id": dataset_id, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            if fail_fast:
                raise
    summary = {
        "cache_dir": str(cache_dir),
        "downloaded": sum(row["status"] == "downloaded" for row in results),
        "manual_action_required": sum(row["status"] == "manual_action_required" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "datasets": results,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "download_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--manual-dir", type=Path, help="Directory containing <dataset_id>.jsonl exports")
    parser.add_argument("--limit", type=int, help="Optional per-dataset smoke-test limit")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    summary = download_all(args.cache_dir, limit=args.limit, manual_dir=args.manual_dir, fail_fast=args.fail_fast)
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
