"""Restore the PAT-Questions text that the public release leaves out.

Run once after cloning, from the repository root:

    uv run python scripts/benchmark/rehydrate_pat.py

It downloads the pinned PAT-Questions March 2024 snapshot (about 6 MB), fills in
every redacted question and template, and checks each restored file against the
SHA-256 of the original. Files that already match are left alone, so running it
twice is harmless. Use ``--pat-dir`` with a local checkout of PAT-Questions at the
pinned commit to work offline, and ``--check`` to verify without writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from river_bench.pat_redaction import MANIFEST_SCHEMA, Redactor, fetch_snapshot, pat_strings, rehydrate_entry

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "pat_redaction_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pat-dir", type=Path, help="Local PAT-Questions checkout at the pinned commit")
    parser.add_argument("--check", action="store_true", help="Verify that every file restores; write nothing")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit(f"{MANIFEST}: unexpected schema {manifest.get('schema')!r}")
    redactor = Redactor(pat_strings(fetch_snapshot(args.pat_dir)))
    for entry in manifest["files"]:
        restored = rehydrate_entry(redactor, ROOT, entry)
        if not args.check:
            (ROOT / entry["path"]).write_bytes(restored)
        print(f"ok  {entry['path']}")
    print(f"{len(manifest['files'])} files restored and verified" if not args.check else "all files verify")


if __name__ == "__main__":
    main()
