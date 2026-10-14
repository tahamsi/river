"""Create an auditable addendum resolving stale pre-adjudication review notes.

This does not edit or re-freeze the corpus. It records that legacy construction
notes are superseded by the completed human adjudication, binds the statement to
the annotation workbook hash, and enumerates every affected example. The named
approver must verify the statement before this file is used in a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation-workbook", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for path in args.fixture
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    flagged: list[dict] = []
    for row in rows:
        metadata = row.get("transformation_metadata") or {}
        note = str(metadata.get("review_note", ""))
        if "require human re-review" not in note.lower():
            continue
        if metadata.get("human_review_status") != "complete":
            raise SystemExit(f"{row['example_id']}: stale note exists but human_review_status is not complete")
        checksum = metadata.get("human_label_record_checksum")
        if not checksum:
            raise SystemExit(f"{row['example_id']}: missing human_label_record_checksum")
        flagged.append(
            {
                "example_id": row["example_id"],
                "human_label_record_checksum": checksum,
                "legacy_review_note": note,
            }
        )

    payload = {
        "schema_version": "river-review-note-resolution-v1",
        "resolution": (
            "The legacy review_note predates the final human adjudication. It is superseded by "
            "human_review_status=complete and the bound human label record checksum; no label was inferred here."
        ),
        "approved_by": args.approved_by,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_workbook": args.annotation_workbook.name,
        "annotation_workbook_sha256": sha256(args.annotation_workbook),
        "fixtures": {path.name: sha256(path) for path in args.fixture},
        "resolved_example_count": len(flagged),
        "resolved_example_ids": sorted(row["example_id"] for row in flagged),
        "records": sorted(flagged, key=lambda row: row["example_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote resolution for {len(flagged)} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
