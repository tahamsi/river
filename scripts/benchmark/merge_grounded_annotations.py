"""Merge source-native base annotations with individually reviewed overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/manual_annotation/base_annotations_1500.jsonl"))
    parser.add_argument("--review-dir", type=Path, default=Path("data/manual_annotation"))
    parser.add_argument("--output", type=Path, default=Path("data/manual_annotation/annotations_merged_1500.jsonl"))
    args = parser.parse_args()

    base = [json.loads(line) for line in args.base.read_text(encoding="utf-8").splitlines() if line.strip()]
    overlays: dict[str, dict] = {}
    for path in sorted(args.review_dir.glob("batch_*_longmem_updates.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = row["source_corpus_id"]
            if source_id in overlays:
                raise ValueError(f"duplicate reviewed overlay for {source_id}")
            overlays[source_id] = row

    additions = []
    for path in sorted(args.review_dir.glob("batch_*_retroactive_corrections.jsonl")):
        additions.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    merged = []
    matched: set[str] = set()
    for row in base:
        overlay = overlays.get(row["source_corpus_id"])
        if overlay is not None:
            source_native = row["source_native_record"]
            row.update(overlay)
            row["source_native_record"] = source_native
            row["annotation_method"] = "source_native_extraction_plus_individual_llm_review"
            row["annotation_status"] = overlay["annotation_status"]
            row["gold_hypothesis"] = row.get("gold_answer") if row.get("gold_state") == "active" else None
            labels = {claim["claim_id"]: claim["gold_label"] for claim in row["claims"]}
            row["claim_gold_labels"] = labels
            row["citation_gold_labels"] = labels
            matched.add(row["source_corpus_id"])
        merged.append(row)
    missing = sorted(set(overlays) - matched)
    if missing:
        raise ValueError(f"review overlays do not match base annotations: {missing}")
    existing_ids = {row["source_corpus_id"] for row in merged}
    for row in additions:
        if row["source_corpus_id"] in existing_ids:
            index = next(i for i, existing in enumerate(merged) if existing["source_corpus_id"] == row["source_corpus_id"])
            merged[index] = row
        else:
            merged.append(row)
            existing_ids.add(row["source_corpus_id"])

    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in merged),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(merged), "individual_llm_reviews": len(matched) + len(additions), "added_corrections": len(additions), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
