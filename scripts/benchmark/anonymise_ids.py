"""Rewrite benchmark splits so that no identifier names the answer.

Applies ``river.identifiers.anonymise_row`` in place to every
``river_bench_*.jsonl`` (and ``review_sample.jsonl``) in each directory given,
and refreshes the output checksums in the manifest. The corpus builders apply
the same transform before writing; this script handles corpora built earlier.
With ``--output`` and a single input file it writes a transformed copy instead.

The transform is idempotent: hashed identifiers are left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from river.identifiers import NAMESPACE, anonymise_row, role_named_ids  # noqa: E402

SPLITS = ("train", "dev", "test")


def rewrite(source: Path, destination: Path) -> tuple[int, int]:
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [anonymise_row(json.loads(line)) for line in lines]
    leftover = sum(bool(role_named_ids(row)) for row in rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return len(rows), leftover


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_directory(directory: Path) -> None:
    files = sorted(directory.glob("river_bench_*.jsonl")) + sorted(directory.glob("review_sample.jsonl"))
    if not files:
        raise SystemExit(f"no benchmark files under {directory}")
    for path in files:
        count, leftover = rewrite(path, path)
        print(f"{path}: {count} rows, {leftover} still carrying a role-named id")
        if leftover:
            raise SystemExit("role-named identifiers survived the rewrite")

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = manifest.get("output_sha256")
    if isinstance(checksums, dict):
        for name in list(checksums):
            if name in SPLITS:
                target = directory / f"river_bench_{name}.jsonl"
            elif name == "review_sample":
                target = directory / "review_sample.jsonl"
            elif name == "combined":
                combined = [p for p in directory.glob("river_bench_*.jsonl") if p.stem.split("_")[-1] not in SPLITS]
                target = combined[0] if len(combined) == 1 else None
            else:
                target = None
            if target is not None and target.exists():
                checksums[name] = sha256(target)
    manifest["identifiers"] = {
        "scheme": NAMESPACE,
        "transform": "river.identifiers.anonymise_row",
        "note": (
            "Hypothesis and evidence ids are hashes of the ids the builder assigned, "
            "which named each item's role in the construction; hypotheses are listed "
            "in hash order. See docs/integrity.md."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{manifest_path}: checksums refreshed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="benchmark directories, or one file with --output")
    parser.add_argument("--output", type=Path, help="write a transformed copy of a single file here")
    args = parser.parse_args()

    if args.output:
        if len(args.paths) != 1 or not args.paths[0].is_file():
            parser.error("--output takes exactly one input file")
        count, leftover = rewrite(args.paths[0], args.output)
        print(f"wrote {count} rows to {args.output}; rows still carrying a role-named id: {leftover}")
        return 1 if leftover else 0

    for directory in args.paths:
        rewrite_directory(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
