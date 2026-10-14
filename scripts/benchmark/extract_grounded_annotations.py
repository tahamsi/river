"""Extract source-grounded base annotations for the five-source RIVER corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INGESTED_AT = "2026-09-01T22:25:39Z"


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _annotation_id(row: dict[str, Any]) -> str:
    return "ann-" + hashlib.sha256(row["river_source_id"].encode()).hexdigest()[:20]


def _base(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_id": _annotation_id(row),
        "source_corpus_id": row["river_source_id"],
        "source_dataset": row["source_dataset"],
        "source_split": row["source_split"],
        "source_row_index": row["source_row_index"],
        "source_file_hash": row["source_file_hash"],
        "annotation_method": "source_native_label_extraction",
        "annotation_status": "automatically_annotated_pending_human_review",
        "human_review_status": "pending",
        "stale_hypothesis_id": None,
        "valid_time": None,
        "valid_time_basis": "not_stated",
        "transaction_time": INGESTED_AT,
        "transaction_time_basis": "source corpus file ingestion timestamp",
        "source_native_record": row["raw_record"],
    }


def _fever(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["raw_record"]
    label = _text(raw.get("label")).upper().replace("_", " ")
    gold_label = {"SUPPORTS": "supported", "REFUTES": "contradicted"}.get(label, "neutral")
    citation = None
    if raw.get("evidence_id") is not None and label != "NOT ENOUGH INFO":
        citation = (
            f"FEVER:{raw.get('evidence_wiki_url')}:evidence:{raw['evidence_id']}:"
            f"sentence:{raw.get('evidence_sentence_id')}"
        )
    out = _base(row)
    out.update({
        "question": _text(raw.get("claim")),
        "gold_answer": gold_label,
        "gold_answer_aliases": [label],
        "gold_state": "abstain" if gold_label == "neutral" else "active",
        "gold_hypothesis": None if gold_label == "neutral" else gold_label,
        "temporal_slice": "not_applicable",
        "claims": [{
            "claim_id": f"fever-{raw.get('id')}",
            "text": _text(raw.get("claim")),
            "gold_label": gold_label,
            "gold_citations": [citation] if citation else [],
        }],
        "supported_tasks": ["insufficient_evidence_nei"] if gold_label == "neutral" else ["order_reversal", "evidence_removal"],
        "candidate_tasks_requiring_llm_review": [],
        "rejected_tasks": [],
    })
    return out


def _pat(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["raw_record"]
    answers = [_text(value) for value in raw.get("text answers", []) if _text(value)]
    question = _text(raw.get("question"))
    lowered = question.lower()
    temporal_slice = "current" if "current" in lowered or "currently" in lowered else (
        "historical" if "previous" in lowered or "last before" in lowered else "not_applicable"
    )
    citation = f"PAT:March2024:{raw.get('_river_source_task')}:{raw.get('uniq_id')}"
    out = _base(row)
    out.update({
        "question": question,
        "gold_answer": answers[0] if answers else None,
        "gold_answer_aliases": answers[1:],
        "gold_state": "active" if answers else "abstain",
        "gold_hypothesis": answers[0] if answers else None,
        "temporal_slice": temporal_slice,
        "valid_time_basis": "March 2024 source snapshot" if temporal_slice == "current" else "ordinal temporal wording only",
        "claims": [{
            "claim_id": f"pat-{raw.get('_river_source_task')}-{raw.get('uniq_id')}",
            "text": f"Answer to '{question}': {answers[0] if answers else ''}",
            "gold_label": "supported",
            "gold_citations": [citation],
        }],
        "supported_tasks": ["order_reversal", "evidence_removal"],
        "candidate_tasks_requiring_llm_review": ["supported_genuine_update", "temporal_evolution"] if temporal_slice in {"current", "historical"} else [],
        "rejected_tasks": ["retroactive_correction", "unsupported_late_assertion", "repeated_weak_assertion"],
    })
    return out


def _wiki(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["raw_record"]
    question = _text(raw.get("question"))
    out = _base(row)
    out.update({
        "question": question,
        "gold_answer": None,
        "gold_answer_aliases": [],
        "gold_state": "disputed",
        "gold_hypothesis": None,
        "temporal_slice": "not_applicable",
        "claims": [
            {
                "claim_id": f"wikicontradict-{raw.get('question_ID')}-a",
                "text": _text(raw.get("answer1")),
                "gold_label": "supported",
                "gold_citations": [f"WikiContradict:{raw.get('question_ID')}:context1"],
            },
            {
                "claim_id": f"wikicontradict-{raw.get('question_ID')}-b",
                "text": _text(raw.get("answer2")),
                "gold_label": "supported",
                "gold_citations": [f"WikiContradict:{raw.get('question_ID')}:context2"],
            },
        ],
        "supported_tasks": ["concurrent_disagreement", "order_reversal", "evidence_removal"],
        "candidate_tasks_requiring_llm_review": ["temporal_evolution"],
        "rejected_tasks": ["unsupported_late_assertion", "repeated_weak_assertion", "retroactive_correction"],
    })
    return out


def _longmem(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["raw_record"]
    question = _text(raw.get("question"))
    answer = _text(raw.get("answer"))
    citations = [f"LongMemEval:session:{value}" for value in raw.get("answer_session_ids", [])]
    question_type = _text(raw.get("question_type"))
    temporal_slice = "timeline" if question_type in {"temporal-reasoning", "knowledge-update"} else "not_applicable"
    candidates = ["supported_genuine_update", "temporal_evolution", "retroactive_correction"] if question_type == "knowledge-update" else []
    out = _base(row)
    out.update({
        "question": question,
        "gold_answer": answer,
        "gold_answer_aliases": [],
        "gold_state": "active",
        "gold_hypothesis": answer,
        "temporal_slice": temporal_slice,
        "claims": [{
            "claim_id": f"longmem-{raw.get('question_id')}",
            "text": f"Answer to '{question}': {answer}",
            "gold_label": "supported",
            "gold_citations": citations,
        }],
        "supported_tasks": ["order_reversal", "evidence_removal"],
        "candidate_tasks_requiring_llm_review": candidates,
        "rejected_tasks": ["unsupported_late_assertion", "repeated_weak_assertion"],
    })
    return out


def _two_wiki(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["raw_record"]
    question = _text(raw.get("question"))
    answer = _text(raw.get("answer"))
    citations = [f"2Wiki:{title}:sentence:{index}" for title, index in raw.get("supporting_facts", [])]
    out = _base(row)
    out.update({
        "question": question,
        "gold_answer": answer,
        "gold_answer_aliases": [],
        "gold_state": "active",
        "gold_hypothesis": answer,
        "temporal_slice": "not_applicable",
        "claims": [{
            "claim_id": f"2wiki-{raw.get('_id')}",
            "text": f"Answer to '{question}': {answer}",
            "gold_label": "supported",
            "gold_citations": citations,
        }],
        "supported_tasks": ["order_reversal", "evidence_removal"],
        "candidate_tasks_requiring_llm_review": [],
        "rejected_tasks": ["supported_genuine_update", "temporal_evolution", "retroactive_correction", "unsupported_late_assertion", "repeated_weak_assertion"],
    })
    return out


EXTRACTORS = {
    "fever": _fever,
    "pat_questions": _pat,
    "wikicontradict": _wiki,
    "longmemeval": _longmem,
    "2wikimultihopqa": _two_wiki,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/river_source_corpus_1500.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/manual_annotation/base_annotations_1500.jsonl"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    annotations = [EXTRACTORS[row["source_dataset"]](row) for row in rows]
    for annotation in annotations:
        labels = {claim["claim_id"]: claim["gold_label"] for claim in annotation["claims"]}
        annotation["claim_gold_labels"] = labels
        annotation["citation_gold_labels"] = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in annotations), encoding="utf-8")
    print(json.dumps({"records": len(annotations), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
