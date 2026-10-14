"""Build the expired_evidence family, which tests valid time against arrival order.

Without evidence that has expired by the target time, valid-time filtering
removes nothing and `river_no_valid_time` cannot change a decision. This family
supplies such evidence. Each pair asks a question with a target time and gives
two statements:

  * the current fact, valid across the target time;
  * a superseded fact, whose validity ended before the target time.

Both come from the source. PAT-Questions records, for the same subject and
relation, the answer that holds now and the answer that held before it.

The two variants of a pair differ only in which statement arrives last. A
system that follows arrival order changes its answer between them; one that
follows valid time does not.

  * Both statements use the same phrasing and tense and differ only in the
    object, so the answer cannot be read from the grammar.
  * PAT-Questions gives no dates, only a snapshot label. The order of the two
    answers is taken from the source; the transition date is templated and is
    marked as such (`valid_time_boundary: templated`) in every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from river.identifiers import anonymise_row

# The PAT snapshot the answers were collected against.
SNAPSHOT = datetime(2024, 3, 1, tzinfo=timezone.utc)
CURRENT_VALID_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
EXPIRED_VALID_START = datetime(2015, 1, 1, tzinfo=timezone.utc)
EXPIRED_VALID_END = datetime(2022, 1, 1, tzinfo=timezone.utc)
ARRIVAL_BASE = datetime(2026, 9, 1, tzinfo=timezone.utc)

# One statement form per relation, used for both the current and the superseded
# fact so the two differ only in the object.
STATEMENTS = {
    "work for": "{subject} works for {object}.",
    "belong to": "{subject} belongs to {object}.",
    "hold": "{subject} holds the position of {object}.",
    "play for": "{subject} plays for {object}.",
}
CURRENT_MARKERS = {
    "work for": "does {subject} work for currently",
    "belong to": "does {subject} belong to currently",
    "hold": "does {subject} hold currently",
    "play for": "does {subject} play for currently",
}
PREVIOUS_MARKERS = {
    "work for": "did {subject} work for before the current",
    "belong to": "did {subject} belong to before the current",
    "hold": "did {subject} hold before the current",
    "play for": "did {subject} play for before the current",
}


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def relation_of(template: str) -> str | None:
    for relation in STATEMENTS:
        if relation in template:
            return relation
    return None


def load_pairs(source: Path) -> list[dict]:
    current: dict[tuple, dict] = {}
    previous: dict[tuple, dict] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        template = row.get("template") or ""
        relation = relation_of(template)
        subject = (row.get("subject") or {}).get("subLabel")
        key_id = (row.get("subject") or {}).get("subject")
        answers = row.get("text answers") or []
        if relation is None or not subject or not key_id or not answers:
            continue
        key = (key_id, relation)
        record = {"subject": subject, "answer": str(answers[0]).strip(), "row": row, "relation": relation}
        if "before the current" in template:
            previous.setdefault(key, record)
        elif "currently" in template:
            current.setdefault(key, record)

    pairs = []
    for key, now in current.items():
        was = previous.get(key)
        if was is None or not was["answer"] or now["answer"] == was["answer"]:
            continue
        pairs.append({"key": key, "current": now, "expired": was})
    return pairs


def build(pair: dict, split: str) -> list[dict]:
    relation = pair["current"]["relation"]
    subject = pair["current"]["subject"]
    current_answer = pair["current"]["answer"]
    expired_answer = pair["expired"]["answer"]

    seed = hashlib.sha256(f"expired:{pair['key']}:{current_answer}:{expired_answer}".encode()).hexdigest()[:16]
    base = f"expired-{seed}"
    query = pair["current"]["row"].get("question") or ""

    h_current = f"{base}:h_current"
    h_expired = f"{base}:h_expired"
    hypotheses = [
        {
            "hypothesis_id": h_current, "subject": subject, "predicate": relation,
            "object": current_answer, "evidence_ids": [f"{base}:e_current"],
        },
        {
            "hypothesis_id": h_expired, "subject": subject, "predicate": relation,
            "object": expired_answer, "evidence_ids": [f"{base}:e_expired"],
        },
    ]

    def evidence(suffix: str, hypothesis_id: str, obj: str, valid_start, valid_end, arrival) -> dict:
        return {
            "evidence_id": f"{base}:{suffix}",
            "source_id": f"pat_questions:{pair['key'][0]}:{suffix}",
            "chunk_id": f"{base}:{suffix}:chunk0",
            "text": STATEMENTS[relation].format(subject=subject, object=obj),
            "subject": subject,
            "predicate": relation,
            "object": obj,
            "hypothesis_id": hypothesis_id,
            "transaction_time": iso(arrival),
            "valid_time": {"start": iso(valid_start), "end": iso(valid_end) if valid_end else None, "confidence": 1.0},
            "entailment": 0.5,
            "contradiction": 0.0,
            "neutral": 0.5,
            "extraction_confidence": 1.0,
            "provenance_reliability": 1.0,
            "independence_group": f"{base}:{suffix}",
        }

    rows = []
    # Variant a: the expired statement arrives last, so arrival order points at
    # the wrong answer. Variant b reverses only the arrival times.
    for variant, expired_last in (("a", True), ("b", False)):
        early, late = ARRIVAL_BASE, ARRIVAL_BASE + timedelta(days=1)
        current_arrival = early if expired_last else late
        expired_arrival = late if expired_last else early
        rows.append({
            "example_id": f"{base}:{variant}",
            "base_example_id": base,
            "pair_id": f"{base}:expired_evidence",
            "session_id": f"{base}:{variant}",
            "dataset": "river_bench_v0",
            "source_dataset": "pat_questions",
            "split": split,
            "intervention": "expired_evidence",
            "query": query,
            "target_time": iso(SNAPSHOT),
            "hypotheses": hypotheses,
            # Presented oldest first, so a system reading list order sees the same
            # thing a system reading transaction_time sees, and the two variants
            # are a genuine reversal rather than the same turn twice.
            "evidence": sorted(
                [
                    evidence("e_current", h_current, current_answer, CURRENT_VALID_START, None, current_arrival),
                    evidence("e_expired", h_expired, expired_answer, EXPIRED_VALID_START, EXPIRED_VALID_END, expired_arrival),
                ],
                key=lambda item: item["transaction_time"],
            ),
            "gold_state": "active",
            "gold_hypothesis_id": h_current,
            "intervention_hypothesis_id": h_expired,
            "transformation_metadata": {
                "annotation_method": "source_native_pat_current_previous_join",
                "annotation_status": "generated_pending_human_review",
                "human_review_status": "pending",
                "task_construction": "expired_validity_against_arrival_order",
                "task_evidence_status": "source_grounded",
                "variant_role": "expired_arrives_last" if expired_last else "expired_arrives_first",
                "stale_hypothesis_id": h_expired,
                "source_corpus_id": "pat_questions/manual.jsonl",
                "source_native_record": {
                    "subject_qid": pair["key"][0],
                    "relation": relation,
                    "current_answer": current_answer,
                    "previous_answer": expired_answer,
                    "current_question": pair["current"]["row"].get("question"),
                    "previous_question": pair["expired"]["row"].get("question"),
                    "snapshot": pair["current"]["row"].get("_river_source_snapshot"),
                },
                "valid_time_boundary": "templated",
                "valid_time_boundary_note": (
                    "PAT-Questions records that one answer superseded the other but "
                    "carries no dates, only a snapshot label. The ordering is "
                    "source-grounded; the transition date is not known and these "
                    "boundaries are templated to place the superseded fact before "
                    "the target time and the current fact across it."
                ),
                "leakage_keys": [f"pat:{pair['key'][0]}:{relation}"],
                "review_instruction": (
                    "Check that the two statements differ only in the answer, that the "
                    "superseded answer is the one the source names as previous, and "
                    "that neither sentence hints at which is current."
                ),
            },
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/pat_questions/manual.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/grounded_expired"))
    parser.add_argument("--pairs", type=int, default=120)
    args = parser.parse_args()

    pairs = load_pairs(args.source)
    pairs.sort(key=lambda item: hashlib.sha256(str(item["key"]).encode()).hexdigest())
    print(f"{len(pairs)} joinable current/previous pairs available")
    pairs = pairs[: args.pairs]

    # Same 4:1:1 proportions as the released splits, assigned per pair so both
    # variants and the leakage key stay on one side of every boundary.
    counts = {"train": int(len(pairs) * 0.667), "dev": int(len(pairs) * 0.167)}
    counts["test"] = len(pairs) - counts["train"] - counts["dev"]

    by_split: dict[str, list[dict]] = defaultdict(list)
    index = 0
    for split in ("train", "dev", "test"):
        for _ in range(counts[split]):
            by_split[split].extend(build(pairs[index], split))
            index += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in by_split.items():
        path = args.output_dir / f"river_bench_{split}.jsonl"
        # Role-named ids are a label leak once written; see river/identifiers.py.
        path.write_text("".join(json.dumps(anonymise_row(row)) + "\n" for row in rows), encoding="utf-8")
        print(f"{split}: {len(rows)} examples ({len(rows) // 2} pairs)")

    relations = defaultdict(int)
    for rows in by_split.values():
        for row in rows:
            relations[row["hypotheses"][0]["predicate"]] += 1
    print("by relation:", dict(relations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
