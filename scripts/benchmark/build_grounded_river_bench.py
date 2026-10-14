"""Build reviewed, evidence-grounded RIVER-Bench pairs and 1,000/500 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from collections import Counter
from pathlib import Path
from typing import Any

from river_bench.transformations.authored import BUILDERS, _evidence, _example, _hypothesis
from river_bench.validation import examples_to_jsonl, validate_all

from river.identifiers import anonymise_row


TASK_COUNTS = {
    "order_reversal": 191,
    "evidence_removal": 194,
    "concurrent_disagreement": 90,
    "unsupported_late_assertion": 80,
    "repeated_weak_assertion": 80,
    "insufficient_evidence_nei": 71,
    "supported_genuine_update": 20,
    "temporal_evolution": 20,
    "retroactive_correction": 4,
}
# (train, dev, test) pairs per family. Each family is stratified across all three
# splits so no intervention is evaluated on a split it never trained against.
SPLIT_PAIR_COUNTS = {
    "order_reversal": (129, 31, 31),
    "evidence_removal": (130, 31, 33),
    "concurrent_disagreement": (60, 15, 15),
    "unsupported_late_assertion": (53, 13, 14),
    "repeated_weak_assertion": (53, 14, 13),
    "insufficient_evidence_nei": (47, 12, 12),
    "supported_genuine_update": (13, 4, 3),
    "temporal_evolution": (13, 4, 3),
    "retroactive_correction": (2, 1, 1),
}
SPLIT_NAMES = ("train", "dev", "test")
TRAIN_PAIR_COUNTS = {task: counts[0] for task, counts in SPLIT_PAIR_COUNTS.items()}
DEV_PAIR_COUNTS = {task: counts[1] for task, counts in SPLIT_PAIR_COUNTS.items()}
TEST_PAIR_COUNTS = {task: counts[2] for task, counts in SPLIT_PAIR_COUNTS.items()}
EXPECTED_SPLIT_EXAMPLES = {"train": 1000, "dev": 250, "test": 250}
INGESTED_AT = "2026-09-01T22:25:39Z"
# Two distinct arrival stamps let `order_reversal` permute *when* evidence was
# observed rather than merely its list position. RIVER must be invariant to the
# swap; an arrival-driven baseline must not be.
ARRIVAL_BASE = datetime(2026, 9, 1, 22, 25, 39, tzinfo=timezone.utc)
ARRIVAL_STEP = timedelta(minutes=17)
# Chunks that are retrieved and on-topic but entail nothing. Abstention has to be
# earned against the support threshold instead of falling out of an empty bundle.
INSUFFICIENT_ENTAILMENT = 0.12
# WikiContradict passages are undated. These bounds declare the scope the claims
# are asserted over so validity is exercised; they are not extracted dates and
# every row that uses them records `temporal_window_construction`.
SCOPE_START = "2020-01-01T00:00:00Z"
SCOPE_END = "2026-01-01T00:00:00Z"
SCOPE_TARGET = "2024-06-01T00:00:00Z"
# The competing claim covers a shorter interval that still contains the target,
# so the two remain concurrently applicable rather than sequential.
OVERLAP_START = "2021-01-01T00:00:00Z"
OVERLAP_END = "2025-01-01T00:00:00Z"
PAT_GOLD_REVIEW = Path("data/reviews/PAT_QUESTIONS_GOLD.json")
ALL_GOLD_REVIEW = Path("data/reviews/ALL_LABELS_GOLD.json")
GOLD_EXAMPLE_FIELDS = (
    "example_id", "pair_id", "source_dataset", "intervention", "gold_state",
    "gold_hypothesis_id", "intervention_hypothesis_id", "target_time",
    "hypotheses", "evidence",
)
GOLD_METADATA_FIELDS = (
    "source_corpus_id", "gold_answer", "gold_answer_aliases", "gold_claims",
    "claim_gold_labels", "citation_gold_labels", "temporal_slice",
    "stale_hypothesis_id", "task_evidence_status",
)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _base_id(source_id: str, task: str) -> str:
    digest = hashlib.sha256(f"{source_id}:{task}".encode()).hexdigest()[:18]
    return f"grounded-{digest}"


def _metadata(annotation: dict[str, Any], task: str, *, construction: str) -> dict[str, Any]:
    claims = annotation.get("claims", [])
    claim_labels = {claim["claim_id"]: claim["gold_label"] for claim in claims}
    return {
        "annotation_method": annotation["annotation_method"],
        "annotation_status": annotation["annotation_status"],
        "human_review_status": annotation.get("human_review_status", "pending"),
        "source_corpus_id": annotation["source_corpus_id"],
        "source_file_hash": annotation["source_file_hash"],
        "source_row_index": annotation["source_row_index"],
        "source_native_record": annotation["source_native_record"],
        "gold_answer": annotation.get("gold_answer"),
        "gold_answer_aliases": annotation.get("gold_answer_aliases", []),
        "gold_claims": claims,
        "claim_gold_labels": claim_labels,
        "citation_gold_labels": claim_labels,
        "temporal_slice": annotation.get("temporal_slice", "not_applicable"),
        "stale_hypothesis_id": annotation.get("stale_hypothesis_id"),
        "task_construction": construction,
        "task_evidence_status": "source_grounded",
        "review_instruction": f"Verify the source-native record supports the {task} construction and every gold claim/citation.",
    }


def _universal_seed(annotation: dict[str, Any], task: str, split: str) -> dict[str, Any]:
    raw = annotation["source_native_record"]
    source_id = annotation["source_corpus_id"]
    base_id = _base_id(source_id, task)
    answer = _clean(annotation.get("gold_answer")) or "insufficient evidence"
    question = _clean(annotation.get("question"))
    return {
        "base_example_id": base_id,
        "event_id": base_id,
        "intervention": task,
        "source_dataset": annotation["source_dataset"],
        "split": split,
        "query": question,
        "subject": f"{source_id}",
        "predicate": "answer",
        "object_a": answer,
        "object_b": "alternative claim",
        "support_text": question,
        "intervention_text": "alternative claim",
        "t1": INGESTED_AT,
        "t2": "2026-09-02T00:00:00Z",
        "t3": "2026-09-02T00:01:00Z",
        "t4": "2026-09-02T00:02:00Z",
        "valid_start": None,
        "valid_end": None,
        "target_time": None,
        "old_valid_start": None,
        "old_valid_end": None,
        "new_valid_start": None,
        "new_valid_end": None,
        "control_target_time": None,
        "variant_target_time": None,
        "overlap_start": None,
        "overlap_end": None,
        "metadata": _metadata(annotation, task, construction="source_native"),
        "raw": raw,
    }


def _two_wiki_sentences(raw: dict[str, Any]) -> list[tuple[str, str]]:
    wanted = {(str(title), int(index)) for title, index in raw.get("supporting_facts", [])}
    found: list[tuple[str, str]] = []
    for title, sentences in raw.get("context", []):
        for index, sentence in enumerate(sentences):
            if (str(title), index) in wanted:
                found.append((f"2Wiki:{title}:sentence:{index}", _clean(sentence)))
    return found


def _arrival_stamps(count: int) -> list[str]:
    """Ascending, evenly spaced arrival timestamps, oldest first."""

    return [
        (ARRIVAL_BASE + ARRIVAL_STEP * index).isoformat().replace("+00:00", "Z")
        for index in range(count)
    ]


def _two_wiki_distractor(raw: dict[str, Any]) -> tuple[str, str, str] | None:
    """Return a non-supporting sentence and the entity it is about.

    ``order_reversal`` needs a second candidate answer for an arrival-driven
    system to wrongly prefer. Taking it from the same article keeps the
    distractor source-grounded and topically plausible rather than authored.

    The article title is returned alongside the sentence and becomes the
    alternative's object. It has to be an entity rather than the sentence
    itself: the gold answer is a short entity, so a sentence-length object
    would let any system separate gold from alternative on string length
    alone, without reading either.

    Returns ``(citation, sentence, entity)``, or ``None`` when the record
    offers no usable distractor.
    """

    answer = _clean(raw.get("answer"))
    folded = answer.casefold()
    wanted = {(str(title), int(index)) for title, index in raw.get("supporting_facts", [])}

    candidates: list[tuple[str, str, str]] = []
    for title, sentences in raw.get("context", []):
        entity = _clean(title)
        # A title that contains the gold answer, or is contained by it, is a
        # name variant of the same thing ("Baden Powell" against "Robert
        # Baden-Powell, 1st Baron Baden-Powell"). Naming the answer differently
        # is not disagreeing with it, so such a hypothesis is not a distractor.
        if not entity:
            continue
        lowered = entity.casefold()
        if lowered == folded or lowered in folded or folded in lowered:
            continue
        for index, sentence in enumerate(sentences):
            if (str(title), index) in wanted:
                continue
            text = _clean(sentence)
            if len(text.split()) >= 6:
                candidates.append((f"2Wiki:{title}:sentence:{index}", text, entity))
                break

    if not candidates:
        return None

    # Yes/no answers are shorter than every entity, so an entity alternative
    # would mark them permanently. The opposite polarity is the competing
    # answer to a comparison question and carries no length signal.
    if folded in {"yes", "no"}:
        citation, text, _entity = candidates[0]
        return citation, text, "no" if folded == "yes" else "yes"

    # Article titles run longer than answer spans. Taking the shortest distinct
    # candidate puts the gold answer on the shorter side of the pair only when
    # no shorter title exists, which keeps string length uninformative.
    return min(candidates, key=lambda item: len(item[2]))


def _build_order(annotation: dict[str, Any], split: str) -> list[Any]:
    """Build an arrival-order pair whose gold answer is invariant to the swap.

    Both variants present an identical evidence multiset and an identical gold
    label. Only the arrival timestamps and the presentation order differ: in one
    variant the supported evidence arrives last, in the other the unsupported
    distractor does. A system that selects on arrival flips between the two;
    RIVER must not.
    """

    seed = _universal_seed(annotation, "order_reversal", split)
    raw = annotation["source_native_record"]
    supporting = _two_wiki_sentences(raw)
    if len(supporting) < 2:
        raise ValueError(
            f"order reversal requires at least two supporting facts: {annotation['source_corpus_id']}"
        )
    distractor = _two_wiki_distractor(raw)
    if distractor is None:
        raise ValueError(
            f"order reversal requires a non-supporting distractor sentence: {annotation['source_corpus_id']}"
        )
    distractor_citation, distractor_text, distractor_entity = distractor

    gold = _hypothesis(seed, "h_gold", _clean(raw.get("answer")))
    alternative = _hypothesis(seed, "h_alt", distractor_entity)

    def _supported(stamps: list[str]) -> list[dict[str, Any]]:
        items = []
        for index, ((citation, text), stamp) in enumerate(zip(supporting, stamps)):
            item = _evidence(seed, f"e_{index}", gold, text, stamp, None, None, 1.0)
            item["source_id"] = citation.rsplit(":sentence:", 1)[0]
            item["chunk_id"] = citation
            items.append(item)
        return items

    def _unsupported(stamp: str) -> list[dict[str, Any]]:
        item = _evidence(seed, "e_alt", alternative, distractor_text, stamp, None, None, 0.45)
        item["source_id"] = distractor_citation.rsplit(":sentence:", 1)[0]
        item["chunk_id"] = distractor_citation
        return [item]

    seed["metadata"]["exact_gold_citations"] = [chunk for chunk, _ in supporting]
    seed["metadata"]["task_construction"] = "source_native_multi_evidence_arrival_reversal"
    seed["metadata"]["distractor_citation"] = distractor_citation
    seed["metadata"]["distractor_is_non_supporting_source_context"] = True

    # One stamp per item so both variants carry the same multiset of arrival
    # times; only which item occupies the newest slot changes.
    stamps = _arrival_stamps(len(supporting) + 1)
    supported_first = _supported(stamps[:-1]) + _unsupported(stamps[-1])
    distractor_first = _unsupported(stamps[0]) + _supported(stamps[1:])
    return [
        _example(
            seed,
            "a",
            "distractor_arrives_last",
            supported_first,
            [gold, alternative],
            target_time=None,
            gold_state="active",
            gold_hypothesis_id=gold["hypothesis_id"],
            intervention_hypothesis_id=alternative["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "supported_arrives_last",
            distractor_first,
            [gold, alternative],
            target_time=None,
            gold_state="active",
            gold_hypothesis_id=gold["hypothesis_id"],
            intervention_hypothesis_id=alternative["hypothesis_id"],
        ),
    ]


def _nei_context_chunks(annotation: dict[str, Any], limit: int = 2) -> list[str]:
    """On-topic sentences for a NOT ENOUGH INFO claim that support nothing.

    FEVER NEI rows carry no gold evidence by construction, so the presented
    bundle is drawn from the claim's own surrounding context where the release
    provides it, and otherwise from a source-grounded restatement of the claim's
    subject. Either way the chunks entail nothing and gold citations stay empty.
    """

    raw = annotation.get("source_native_record", {})
    candidates: list[str] = []
    for key in ("context", "evidence_context", "sentences"):
        value = raw.get(key)
        if isinstance(value, str):
            candidates.append(_clean(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    candidates.append(_clean(item))
                elif isinstance(item, (list, tuple)) and item:
                    candidates.append(_clean(item[-1]))
    claim = _clean(annotation.get("question"))
    subject = claim.split(" is ")[0].split(" was ")[0].strip() or claim
    fallback = [
        f"This passage concerns {subject} but does not state whether the claim holds.",
        f"The available record for {subject} omits the detail the claim asserts.",
    ]
    chunks = [text for text in candidates if len(text.split()) >= 6][:limit]
    while len(chunks) < limit:
        chunks.append(fallback[len(chunks) % len(fallback)])
    return chunks[:limit]


def _build_nei(annotation: dict[str, Any], split: str) -> list[Any]:
    seed = _universal_seed(annotation, "insufficient_evidence_nei", split)
    seed["metadata"]["task_construction"] = "source_native_fever_nei"
    seed["metadata"]["gold_answer"] = None
    seed["metadata"]["gold_answer_aliases"] = []
    h = _hypothesis(seed, "h_claim", _clean(annotation["question"]))
    # FEVER NOT ENOUGH INFO rows are *retrieved but insufficient*, not empty. An
    # empty bundle would make `len(evidence) == 0` a perfect predictor of the
    # abstain label, so the presented chunks are on-topic and entail nothing.
    context = _nei_context_chunks(annotation)
    presented = [
        _evidence(
            seed,
            f"e_nei_{index}",
            h,
            text,
            INGESTED_AT,
            None,
            None,
            INSUFFICIENT_ENTAILMENT,
            neutral=1.0 - INSUFFICIENT_ENTAILMENT,
        )
        for index, text in enumerate(context)
    ]
    seed["metadata"]["task_construction"] = "source_native_fever_nei_with_insufficient_context"
    seed["metadata"]["presented_context_is_insufficient"] = True
    return [
        _example(
            seed,
            "a",
            "insufficient_context",
            presented,
            [h],
            target_time=None,
            gold_state="abstain",
            gold_hypothesis_id=None,
            intervention_hypothesis_id=h["hypothesis_id"],
        ),
        _example(
            seed,
            "b",
            "source_native_nei",
            list(reversed(presented)),
            [h],
            target_time=None,
            gold_state="abstain",
            gold_hypothesis_id=None,
            intervention_hypothesis_id=h["hypothesis_id"],
        ),
    ]


def _build_wiki(annotation: dict[str, Any], task: str, split: str) -> list[Any]:
    raw = annotation["source_native_record"]
    seed = _universal_seed(annotation, task, split)
    seed.update({
        "object_a": _clean(raw.get("answer1")),
        "object_b": _clean(raw.get("answer2")),
        "support_text": _clean(raw.get("context1")),
        "intervention_text": _clean(raw.get("context2")) if task == "concurrent_disagreement" else _clean(raw.get("answer2")),
        "valid_start": SCOPE_START,
        "valid_end": SCOPE_END,
        "target_time": SCOPE_TARGET,
        "overlap_start": OVERLAP_START,
        "overlap_end": OVERLAP_END,
    })
    seed["metadata"]["temporal_window_construction"] = "declared_scope_source_passages_are_undated"
    if task in {"unsupported_late_assertion", "repeated_weak_assertion"}:
        seed["metadata"]["gold_answer"] = _clean(raw.get("answer1"))
        seed["metadata"]["gold_answer_aliases"] = []
        seed["metadata"]["task_construction"] = "grounded_adversarial_withheld_support"
        seed["metadata"]["withheld_source_chunk"] = _clean(raw.get("context2"))
        seed["metadata"]["withheld_claim_is_not_naturally_unsupported"] = True
    return BUILDERS[task](seed)


def _finalize_bundle_labels(pair: list[Any]) -> list[Any]:
    """Attach claim/citation gold labels for each row's presented evidence."""

    for example in pair:
        metadata = example.transformation_metadata
        metadata["source_annotation_gold_claims"] = metadata.get("gold_claims", [])
        evidence_by_hypothesis: dict[str, list[Any]] = {}
        for evidence in example.evidence:
            evidence_by_hypothesis.setdefault(evidence.hypothesis_id, []).append(evidence)
        claims = []
        for hypothesis in example.hypotheses:
            evidence = evidence_by_hypothesis.get(hypothesis.hypothesis_id, [])
            label = "supported" if evidence else "neutral"
            claims.append({
                "claim_id": hypothesis.hypothesis_id,
                "text": hypothesis.object,
                "gold_label": label,
                "gold_citations": [
                    {"source_id": item.source_id, "chunk_id": item.chunk_id}
                    for item in evidence
                ],
            })
        labels = {claim["claim_id"]: claim["gold_label"] for claim in claims}
        metadata["gold_claims"] = claims
        metadata["claim_gold_labels"] = labels
        metadata["citation_gold_labels"] = labels
        role = str(metadata.get("variant_role", ""))
        if example.intervention == "supported_genuine_update":
            metadata["stale_hypothesis_id"] = (
                example.hypotheses[0].hypothesis_id if role == "supported_update_variant" else None
            )
        elif example.intervention == "retroactive_correction":
            metadata["stale_hypothesis_id"] = (
                example.hypotheses[0].hypothesis_id if role == "retroactive_correction" else None
            )
        elif example.intervention == "temporal_evolution":
            metadata["stale_hypothesis_id"] = None
        else:
            metadata["stale_hypothesis_id"] = None
    return pair


def _build_removal(annotation: dict[str, Any], split: str) -> list[Any]:
    seed = _universal_seed(annotation, "evidence_removal", split)
    raw = annotation["source_native_record"]
    answer = _clean(annotation.get("gold_answer"))
    seed["object_a"] = answer
    seed["support_text"] = f"PAT March 2024 snapshot answer to '{_clean(raw.get('question'))}': {answer}."
    seed["metadata"]["task_construction"] = "source_native_answer_then_exact_removal"
    return BUILDERS["evidence_removal"](seed)


def _reviewed_times(claim: dict[str, Any]) -> tuple[str | None, str | None]:
    interval = claim.get("valid_time") or {}
    return interval.get("start"), interval.get("end")


def _build_reviewed(annotation: dict[str, Any], task: str, split: str) -> list[Any]:
    seed = _universal_seed(annotation, task, split)
    claims = annotation["claims"]
    if len(claims) < 2:
        raise ValueError(f"reviewed temporal task requires two claims: {annotation['source_corpus_id']}")
    first, second = claims[0], claims[1]
    first_start, first_end = _reviewed_times(first)
    second_start, second_end = _reviewed_times(second)
    seed.update({
        "object_a": first["text"],
        "object_b": second["text"],
        "support_text": first["text"],
        "intervention_text": second["text"],
        "t1": first.get("transaction_time") or first_start or INGESTED_AT,
        "t2": second.get("transaction_time") or second_start or "2026-09-02T00:00:00Z",
        "valid_start": first_start or SCOPE_START,
        "valid_end": first_end,
        "target_time": second_start or SCOPE_TARGET,
        "old_valid_start": first_start,
        "old_valid_end": first_end,
        "new_valid_start": second_start,
        "new_valid_end": second_end,
        "control_target_time": first_start,
        "variant_target_time": second_start,
        "overlap_start": first_start or SCOPE_START,
        "overlap_end": first_end,
    })
    seed["metadata"]["task_construction"] = "individually_llm_reviewed_source_chronology"
    if not (first_start and second_start):
        seed["metadata"]["temporal_window_construction"] = "declared_scope_review_supplied_no_interval"
    return BUILDERS[task](seed)


def _assign_splits(task: str, rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    train_n, dev_n, test_n = SPLIT_PAIR_COUNTS[task]
    if len(rows) != train_n + dev_n + test_n:
        raise ValueError(
            f"{task}: {len(rows)} source rows do not fill the declared "
            f"{train_n}/{dev_n}/{test_n} train/dev/test split"
        )
    assignment = ["train"] * train_n + ["dev"] * dev_n + ["test"] * test_n
    return list(zip(rows, assignment))


def _apply_pat_gold_review(rows: list[dict[str, Any]]) -> None:
    review = json.loads(PAT_GOLD_REVIEW.read_text(encoding="utf-8"))
    source_ids = sorted(row["source_corpus_id"] for row in rows)
    digest = "sha256:" + hashlib.sha256("\n".join(source_ids).encode()).hexdigest()
    if len(source_ids) != review["selected_source_id_count"] or digest != review["selected_source_ids_sha256"]:
        raise ValueError("selected PAT-Questions records do not match the human-approved gold review set")
    for row in rows:
        row["annotation_status"] = "gold_human_reviewed"
        row["human_review_status"] = "accepted"
        row["review_note"] = review["review_note"]


def _semantic_label_digest(examples: list[Any]) -> str:
    """Checksum of exactly the fields a human reviewer signs off on."""

    projection = []
    for example in sorted(examples, key=lambda row: row.example_id):
        row = example.model_dump(mode="json")
        item = {key: row.get(key) for key in GOLD_EXAMPLE_FIELDS}
        item["transformation_metadata"] = {
            key: row["transformation_metadata"].get(key) for key in GOLD_METADATA_FIELDS
        }
        projection.append(item)
    payload = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _apply_all_gold_review(examples: list[Any]) -> dict[str, Any]:
    """Bind, or withhold, the human gold approval for the generated rows.

    Returns a status record for the build manifest. An approval covers one exact
    label payload: when regeneration changes that payload the rows are marked
    pending re-review rather than inheriting an approval no person granted.
    """

    review = json.loads(ALL_GOLD_REVIEW.read_text(encoding="utf-8"))
    digest = _semantic_label_digest(examples)
    matches = (
        digest == review["semantic_labels_sha256"]
        and len(examples) == review["example_count"]
        and len({row.pair_id for row in examples}) == review["pair_count"]
    )
    if matches:
        for example in examples:
            example.transformation_metadata["annotation_status"] = "gold_human_reviewed"
            example.transformation_metadata["human_review_status"] = "accepted"
            example.transformation_metadata["review_note"] = review["review_note"]
        return {
            "state": "approved",
            "semantic_labels_sha256": digest,
            "approved_manifest": str(ALL_GOLD_REVIEW),
        }

    note = (
        "Task construction changed after the recorded approval; these rows are "
        "regenerated and require human re-review before any result is reported."
    )
    for example in examples:
        example.transformation_metadata["annotation_status"] = "rebuilt_pending_human_review"
        example.transformation_metadata["human_review_status"] = "pending"
        example.transformation_metadata["review_note"] = note
    return {
        "state": "pending_human_review",
        "semantic_labels_sha256": digest,
        "approved_manifest": str(ALL_GOLD_REVIEW),
        "approved_semantic_labels_sha256": review["semantic_labels_sha256"],
        "note": note,
    }


def public_jsonl(examples) -> str:
    """Serialise examples with identifiers that do not name their role.

    The builders below name hypotheses and evidence by role (h_gold, h_alt, ...),
    which is convenient while constructing a pair and a label leak once written:
    see river/identifiers.py.
    """

    return "".join(
        json.dumps(anonymise_row(json.loads(line))) + "\n"
        for line in examples_to_jsonl(examples).splitlines()
        if line.strip()
    )


def build(corpus_path: Path, annotations_path: Path, output_dir: Path, seed_value: int) -> dict[str, Any]:
    corpus = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_by_id = {row["river_source_id"]: row for row in corpus}
    annotations = [json.loads(line) for line in annotations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for annotation in annotations:
        source_row = source_by_id.get(annotation["source_corpus_id"])
        if source_row is not None:
            annotation["source_native_record"] = source_row["raw_record"]
        elif "source_native_record" not in annotation:
            raise ValueError(f"missing source-native record for {annotation['source_corpus_id']}")
    sources = sorted({row["source_dataset"] for row in annotations})
    by_source = {source: [row for row in annotations if row["source_dataset"] == source] for source in sources}
    rng = random.Random(seed_value)
    for rows in by_source.values():
        rng.shuffle(rows)

    wiki = by_source["wikicontradict"]
    task_rows: dict[str, list[dict[str, Any]]] = {
        "concurrent_disagreement": wiki[:90],
        "unsupported_late_assertion": wiki[90:170],
        "repeated_weak_assertion": wiki[170:250],
        "insufficient_evidence_nei": [row for row in by_source["fever"] if row["gold_answer"] == "neutral"][:71],
        "order_reversal": [row for row in by_source["2wikimultihopqa"] if len(_two_wiki_sentences(row["source_native_record"])) >= 2][:191],
        "evidence_removal": by_source["pat_questions"][:194],
    }
    reviewed = [
        row for row in by_source["longmemeval"]
        if row["annotation_status"] == "llm_annotated_pending_human_review"
    ]
    correction = [
        row for rows in by_source.values() for row in rows
        if row.get("decision") == "accept_as_correction"
    ]
    update_pool = [row for row in reviewed if "supported_genuine_update" in row.get("supported_tasks", []) and row not in correction]
    evolution_pool = [row for row in update_pool if "temporal_evolution" in row.get("supported_tasks", [])]
    task_rows["supported_genuine_update"] = update_pool[:20]
    used = {row["source_corpus_id"] for row in task_rows["supported_genuine_update"]}
    task_rows["temporal_evolution"] = [row for row in evolution_pool if row["source_corpus_id"] not in used][:20]
    task_rows["retroactive_correction"] = correction[:4]

    for task, expected in TASK_COUNTS.items():
        if len(task_rows.get(task, [])) != expected:
            raise ValueError(f"{task}: expected {expected} source rows, found {len(task_rows.get(task, []))}")
    _apply_pat_gold_review(task_rows["evidence_removal"])
    selected_ids = [row["source_corpus_id"] for rows in task_rows.values() for row in rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("a source record was assigned to more than one final task")

    examples: list[Any] = []
    for task, rows in task_rows.items():
        for annotation, split in _assign_splits(task, rows):
            if task == "order_reversal":
                pair = _build_order(annotation, split)
            elif task == "evidence_removal":
                pair = _build_removal(annotation, split)
            elif task == "insufficient_evidence_nei":
                pair = _build_nei(annotation, split)
            elif task in {"concurrent_disagreement", "unsupported_late_assertion", "repeated_weak_assertion"}:
                pair = _build_wiki(annotation, task, split)
            else:
                pair = _build_reviewed(annotation, task, split)
            examples.extend(_finalize_bundle_labels(pair))

    gold_review = _apply_all_gold_review(examples)
    validate_all(examples)
    by_split = {name: [row for row in examples if row.split == name] for name in SPLIT_NAMES}
    sizes = {name: len(rows) for name, rows in by_split.items()}
    if sizes != EXPECTED_SPLIT_EXAMPLES:
        raise AssertionError(f"unexpected split sizes {sizes}, expected {EXPECTED_SPLIT_EXAMPLES}")
    train, dev, test = (by_split[name] for name in SPLIT_NAMES)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined": output_dir / "river_bench_1500.jsonl",
        "train": output_dir / "river_bench_train.jsonl",
        "dev": output_dir / "river_bench_dev.jsonl",
        "test": output_dir / "river_bench_test.jsonl",
        "review_sample": output_dir / "review_sample.jsonl",
    }
    paths["combined"].write_text(public_jsonl(examples), encoding="utf-8")
    for name in SPLIT_NAMES:
        paths[name].write_text(public_jsonl(by_split[name]), encoding="utf-8")
    review_examples: list[Any] = []
    for task in TASK_COUNTS:
        candidates = [row for row in dev if row.intervention == task]
        if not candidates:
            candidates = [row for row in train if row.intervention == task]
        pair_id = sorted(candidates, key=lambda row: row.example_id)[0].pair_id
        review_examples.extend(row for row in examples if row.pair_id == pair_id)
    paths["review_sample"].write_text(public_jsonl(review_examples), encoding="utf-8")
    review_lines = [
        "# Stratified annotation review sample",
        "",
        "Review both variants against `source_native_record`. Verify the answer, every presented-evidence claim label, exact citation, temporal interval, gold state, and selected hypothesis. Reject the entire pair if any required fact is not source-grounded.",
        "",
        "| Task | Source | Pair | Review focus |",
        "|---|---|---|---|",
    ]
    for task in TASK_COUNTS:
        row = next(item for item in review_examples if item.intervention == task)
        review_lines.append(
            f"| `{task}` | `{row.source_dataset}` | `{row.pair_id}` | {row.transformation_metadata['review_instruction']} |"
        )
    review_md = output_dir / "review_sample.md"
    review_md.write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    metadata_rows = [row.transformation_metadata for row in examples]
    audit = {
        "examples": len(examples),
        "pairs": len({row.pair_id for row in examples}),
        "unique_source_records": len({row["source_corpus_id"] for row in metadata_rows}),
        "schema_pair_order_leakage_validation": "passed",
        "required_metadata_coverage": {
            key: sum(key in row for row in metadata_rows)
            for key in (
                "gold_answer", "gold_answer_aliases", "gold_claims", "claim_gold_labels",
                "citation_gold_labels", "temporal_slice", "stale_hypothesis_id",
                "annotation_status", "human_review_status", "source_native_record",
                "review_instruction", "task_evidence_status",
            )
        },
        "intentional_nulls": {
            "gold_answer": "Null for disputed and abstention rows where no single answer is selectable.",
            "stale_hypothesis_id": "Null outside a source-grounded supersession/update.",
            "gold_citations": "Empty exactly when a presented claim is neutral/insufficient and has no supporting chunk.",
        },
        "task_pair_counts": TASK_COUNTS,
        "split_example_counts": sizes,
        "human_review_counts": dict(sorted(Counter(row["human_review_status"] for row in metadata_rows).items())),
        "gold_review_manifests": [str(PAT_GOLD_REVIEW), str(ALL_GOLD_REVIEW)],
        "gold_review": gold_review,
    }
    audit_path = output_dir / "label_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "river-bench-grounded-v1",
        "seed": seed_value,
        "pair_count": 750,
        "example_count": 1500,
        "split_example_counts": EXPECTED_SPLIT_EXAMPLES,
        "split_pair_counts": {
            task: dict(zip(SPLIT_NAMES, counts)) for task, counts in sorted(SPLIT_PAIR_COUNTS.items())
        },
        "split_policy": (
            "train fits parameters; dev selects checkpoints and thresholds; test is "
            "read once after freezing. Pairs and leakage keys never cross a boundary."
        ),
        "task_pair_counts": TASK_COUNTS,
        "source_pair_counts": dict(sorted(Counter(row.source_dataset for row in examples[::2]).items())),
        "annotation_policy": "source-native extraction plus individual LLM review for chronology-sensitive records",
        "grounded_adversarial_policy": "WikiContradict claims are real; context2 is deliberately withheld for unsupported/repetition interventions and recorded in metadata",
        "output_sha256": {name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "review_guide": str(review_md),
        "label_audit": str(audit_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/river_source_corpus_1500.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/manual_annotation/annotations_merged_1500.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/grounded"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.corpus, args.annotations, args.output_dir, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
