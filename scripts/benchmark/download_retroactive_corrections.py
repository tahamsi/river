"""Download and annotate a small, pinned set of explicit corrections."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


WIKI_REVISIONS = (
    (1372716678, 1372717232),  # Norwood News closure date
    (1372311485, 1372717212),  # Frank Dillane co-star
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_compare(old_revision: int, new_revision: int) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "action": "compare", "format": "json", "fromrev": old_revision,
        "torev": new_revision,
        "prop": "title|ids|timestamp|user|comment|diff|size",
    })
    request = urllib.request.Request(
        "https://en.wikipedia.org/w/api.php?" + params,
        headers={"User-Agent": "RIVER-bench-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _plain_diff(payload: dict[str, Any]) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", payload["compare"]["*"])).split())


def _annotation_base(source_id: str, source: str, source_hash: str, row: int, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_id": "ann-" + hashlib.sha256(source_id.encode()).hexdigest()[:20],
        "source_corpus_id": source_id,
        "source_dataset": source,
        "source_split": "pinned_explicit_corrections",
        "source_row_index": row,
        "source_file_hash": source_hash,
        "annotation_method": "individual_llm_review_of_explicit_source_correction",
        "annotation_status": "llm_annotated_pending_human_review",
        "human_review_status": "pending",
        "decision": "accept_as_correction",
        "supported_tasks": ["retroactive_correction"],
        "candidate_tasks_requiring_llm_review": [],
        "rejected_tasks": ["supported_genuine_update", "temporal_evolution"],
        "gold_state": "active",
        "temporal_slice": "historical",
        "source_native_record": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longmem", type=Path, default=Path("data/longmemeval/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/retroactive_corrections"))
    parser.add_argument("--annotation-output", type=Path, default=Path("data/manual_annotation/batch_006_retroactive_corrections.jsonl"))
    parser.add_argument("--wiki-cache-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wiki_payloads = []
    for old_revision, new_revision in WIKI_REVISIONS:
        cached = args.wiki_cache_dir / f"wiki_diff_{new_revision}.json" if args.wiki_cache_dir else None
        output_cache = args.output_dir / f"wikipedia_compare_{old_revision}_{new_revision}.json"
        if cached and cached.exists():
            payload = json.loads(cached.read_text())
        elif output_cache.exists():
            payload = json.loads(output_cache.read_text())
        else:
            payload = _fetch_compare(old_revision, new_revision)
        output_cache.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        wiki_payloads.append(payload)

    longmem_rows = [json.loads(line) for line in args.longmem.read_text(encoding="utf-8").splitlines()]
    longmem = longmem_rows[458]
    if longmem.get("question_id") != "77eafa52":
        raise ValueError("pinned LongMemEval correction record moved")
    longmem_path = args.output_dir / "longmemeval_77eafa52.json"
    longmem_path.write_text(json.dumps(longmem, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    annotations: list[dict[str, Any]] = []
    source_id = "longmemeval:77eafa52:row:458"
    row = _annotation_base(source_id, "longmemeval", _sha(longmem_path), 458, longmem)
    row.update({
        "question": longmem["question"], "gold_answer": "$2,800", "gold_answer_aliases": ["2800 dollars", "$300 more"],
        "gold_hypothesis": "$2,800", "stale_hypothesis_id": "trip-price-2500",
        "claims": [
            {"claim_id": "trip-price-original", "text": "The quoted trip price was $2,500.", "gold_label": "supported", "gold_citations": ["LongMemEval:session:answer_33c251f0_1"], "transaction_time": "2023-05-20T02:22:00", "valid_time": {"start": None, "end": None}},
            {"claim_id": "trip-price-corrected", "text": "The corrected trip price was $2,800.", "gold_label": "supported", "gold_citations": ["LongMemEval:session:answer_33c251f0_2"], "transaction_time": "2023-05-23T19:38:00", "valid_time": {"start": None, "end": None}},
        ],
        "review_note": "The later answer session explicitly calls $2,800 the corrected price for the same trip previously quoted at $2,500.",
    })
    annotations.append(row)

    specs = [
        {
            "old": 1372716678, "new": 1372717232, "question": "When did the Norwood News publish its final issue?",
            "answer": "August 28, 2026", "aliases": ["Aug. 28, 2026"], "stale": "norwood-final-2026-09-01",
            "old_text": "The Norwood News published its final issue on September 1, 2026.",
            "new_text": "The corrected final-issue date for the Norwood News was August 28, 2026.",
            "valid": "2026-08-28T00:00:00Z",
        },
        {
            "old": 1372311485, "new": 1372717212, "question": "Who plays Marianne opposite Frank Dillane's John Willoughby in the 2026 Sense and Sensibility adaptation?",
            "answer": "Esmé Creed-Miles", "aliases": ["Esme Creed-Miles"], "stale": "marianne-daisy-edgar-jones",
            "old_text": "The article identified Daisy Edgar-Jones as Marianne opposite Frank Dillane's John Willoughby.",
            "new_text": "The corrected article identifies Esmé Creed-Miles as Marianne opposite Frank Dillane's John Willoughby.",
            "valid": "2026-01-01T00:00:00Z",
        },
    ]
    for index, (payload, spec) in enumerate(zip(wiki_payloads, specs), start=1):
        compare = payload["compare"]
        if compare["fromrevid"] != spec["old"] or compare["torevid"] != spec["new"]:
            raise ValueError("Wikipedia compare response did not match pinned revisions")
        raw = {"compare": {k: v for k, v in compare.items() if k != "*"}, "plain_diff": _plain_diff(payload), "license": "CC BY-SA 4.0", "source_url": f"https://en.wikipedia.org/?diff={spec['new']}&oldid={spec['old']}"}
        raw_path = args.output_dir / f"wikipedia_correction_{spec['new']}.json"
        raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        source_id = f"wikipedia_revision:{spec['old']}:{spec['new']}"
        row = _annotation_base(source_id, "wikipedia_revision", _sha(raw_path), index, raw)
        row.update({
            "question": spec["question"], "gold_answer": spec["answer"], "gold_answer_aliases": spec["aliases"],
            "gold_hypothesis": spec["answer"], "stale_hypothesis_id": spec["stale"],
            "claims": [
                {"claim_id": f"wiki-{spec['old']}-original", "text": spec["old_text"], "gold_label": "supported", "gold_citations": [f"Wikipedia:revision:{spec['old']}"], "transaction_time": compare["fromtimestamp"], "valid_time": {"start": spec["valid"], "end": None}},
                {"claim_id": f"wiki-{spec['new']}-corrected", "text": spec["new_text"], "gold_label": "supported", "gold_citations": [f"Wikipedia:revision:{spec['new']}"], "transaction_time": compare["totimestamp"], "valid_time": {"start": spec["valid"], "end": None}},
            ],
            "review_note": f"Revision comment explicitly identifies a correction: {compare['tocomment']}",
        })
        annotations.append(row)

    for row in annotations:
        labels = {claim["claim_id"]: claim["gold_label"] for claim in row["claims"]}
        row["claim_gold_labels"] = labels
        row["citation_gold_labels"] = labels
    args.annotation_output.parent.mkdir(parents=True, exist_ok=True)
    args.annotation_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations), encoding="utf-8")
    print(json.dumps({"records": len(annotations), "output": str(args.annotation_output)}, indent=2))


if __name__ == "__main__":
    main()
