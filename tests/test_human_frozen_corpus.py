import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "grounded_human_frozen"


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_release_is_complete_pair_safe_and_checksum_bound():
    manifest = json.loads((DATA / "manifest.json").read_text())
    combined = rows(DATA / "river_bench_1502.jsonl")
    assert len(combined) == 1502
    assert len({row["example_id"] for row in combined}) == 1502
    assert set(Counter(row["pair_id"] for row in combined).values()) == {2}
    assert all(row["transformation_metadata"]["annotation_status"] == "human_adjudicated_frozen" for row in combined)
    assert all(row["transformation_metadata"]["human_review_status"] == "complete" for row in combined)
    redacted = []
    for filename, expected in manifest["output_sha256"].items():
        if b"@@PAT:" in (DATA / filename).read_bytes():
            redacted.append(filename)
            continue
        assert digest(DATA / filename) == expected
    if redacted:
        pytest.skip(f"PAT-Questions text not restored in {redacted}; run scripts/benchmark/rehydrate_pat.py first")


def test_frozen_splits_are_disjoint_and_match_combined_release():
    split_rows = {split: rows(DATA / f"river_bench_{split}.jsonl") for split in ("train", "dev", "test")}
    ids = {split: {row["example_id"] for row in values} for split, values in split_rows.items()}
    assert not (ids["train"] & ids["dev"] or ids["train"] & ids["test"] or ids["dev"] & ids["test"])
    assert sum(map(len, ids.values())) == 1502
    assert all(row["split"] == split for split, values in split_rows.items() for row in values)


def test_gold_state_and_hypothesis_are_coherent():
    for row in rows(DATA / "river_bench_1502.jsonl"):
        hypothesis_ids = {item["hypothesis_id"] for item in row["hypotheses"]}
        if row["gold_state"] == "active":
            assert row["gold_hypothesis_id"] in hypothesis_ids
        else:
            assert row["gold_hypothesis_id"] is None
