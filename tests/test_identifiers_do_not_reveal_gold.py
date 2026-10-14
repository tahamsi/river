"""No identifier or list position may reveal the gold answer.

The corpus builders originally named hypotheses by role (h_gold and h_alt,
h_current and h_expired) and always listed the gold hypothesis first. Because
the answer prompt included these ids, a generator could read the label, and a
tie broken by position picked the gold. After hashing, naive RAG's answer
accuracy fell from 0.876 to 0.684 (see docs/integrity.md).
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from river.identifiers import HASHED, anonymise_row, hashed_id, role_named_ids

CORPORA = [
    Path(os.environ.get("RIVER_GROUNDED_DIR", "data/grounded_nli")),
    Path(os.environ.get("RIVER_RELEASED_DIR", "data/grounded_human_frozen")),
]


def _rows(directory: Path) -> list[dict]:
    paths = [directory / f"river_bench_{split}.jsonl" for split in ("train", "dev", "test")]
    if not all(path.exists() for path in paths):
        pytest.skip(f"benchmark splits not present under {directory}")
    return [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("directory", CORPORA, ids=str)
def test_no_identifier_names_a_role(directory):
    offenders = [(row["example_id"], role_named_ids(row)[:3]) for row in _rows(directory) if role_named_ids(row)]
    assert not offenders, f"{len(offenders)} rows carry role-named ids, e.g. {offenders[:3]}"


@pytest.mark.parametrize("directory", CORPORA, ids=str)
def test_every_identifier_is_hashed(directory):
    """Stronger than the role-name check: it cannot miss a role nobody listed."""

    unhashed = Counter()
    for row in _rows(directory):
        ids = [h.get("hypothesis_id") for h in row.get("hypotheses") or []]
        ids += [e.get("evidence_id") for e in row.get("evidence") or []]
        ids += [e.get("hypothesis_id") for e in row.get("evidence") or []]
        ids += [row.get("gold_hypothesis_id"), row.get("intervention_hypothesis_id")]
        for identifier in filter(None, ids):
            local = identifier.rpartition(":")[2]
            if not HASHED.match(local):
                unhashed[local] += 1
    assert not unhashed, f"unhashed identifiers: {dict(unhashed.most_common(5))}"


@pytest.mark.parametrize("directory", CORPORA, ids=str)
def test_gold_hypothesis_position_varies_within_every_family(directory):
    positions: dict[str, Counter] = defaultdict(Counter)
    for row in _rows(directory):
        ids = [h.get("hypothesis_id") for h in row.get("hypotheses") or []]
        gold = row.get("gold_hypothesis_id")
        if gold in ids and len(ids) >= 2:
            positions[row["intervention"]][ids.index(gold)] += 1
    fixed = {family: dict(counts) for family, counts in positions.items() if sum(counts.values()) >= 20 and len(counts) == 1}
    assert not fixed, f"gold hypothesis always at the same list position: {fixed}"


@pytest.mark.parametrize("directory", CORPORA, ids=str)
def test_paired_examples_still_share_hypothesis_ids(directory):
    """Paired metrics compare hypothesis ids across a pair, so the hash must agree."""

    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in _rows(directory):
        if row.get("pair_id"):
            by_pair[row["pair_id"]].append(row)
    mismatched = [
        pair for pair, rows in by_pair.items()
        if len({frozenset(h["hypothesis_id"] for h in row.get("hypotheses") or []) for row in rows}) > 1
    ]
    assert not mismatched, f"{len(mismatched)} pairs disagree on hypothesis ids, e.g. {mismatched[:3]}"


def test_transform_is_stable_idempotent_and_keeps_pairs_aligned():
    base = {
        "example_id": "x:a", "pair_id": "x", "intervention": "order_reversal", "query": "who h_gold?",
        "gold_hypothesis_id": "x:h_gold",
        "hypotheses": [
            {"hypothesis_id": "x:h_gold", "evidence_ids": ["x:e_0"]},
            {"hypothesis_id": "x:h_alt", "evidence_ids": ["x:e_alt"]},
        ],
        "evidence": [
            {"evidence_id": "x:e_0", "hypothesis_id": "x:h_gold", "chunk_id": "x:e_0:chunk0", "text": "the h_gold text", "source_id": "src:e_current"},
            {"evidence_id": "x:e_alt", "hypothesis_id": "x:h_alt", "chunk_id": "x:e_alt:chunk0", "text": "t"},
        ],
    }
    once = anonymise_row(base)
    assert anonymise_row(once) == once
    assert not role_named_ids(once)
    assert once["gold_hypothesis_id"] == hashed_id("x:h_gold", "h")
    assert once["evidence"][0]["hypothesis_id"] == once["gold_hypothesis_id"]
    assert once["evidence"][0]["chunk_id"] == hashed_id("x:e_0", "e") + ":chunk0"
    assert once["evidence"][0]["text"] == "the h_gold text" and once["query"] == "who h_gold?"
    assert [e["evidence_id"] for e in once["evidence"]] == [hashed_id("x:e_0", "e"), hashed_id("x:e_alt", "e")]
    assert once["hypotheses"] == sorted(once["hypotheses"], key=lambda h: h["hypothesis_id"])
    partner = anonymise_row({**base, "example_id": "x:b", "evidence": list(reversed(base["evidence"]))})
    assert {h["hypothesis_id"] for h in partner["hypotheses"]} == {h["hypothesis_id"] for h in once["hypotheses"]}
