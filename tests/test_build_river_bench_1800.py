import json
from pathlib import Path

import pytest

from scripts.benchmark.build_river_bench_1800 import SOURCE_QUOTAS, build
from river_bench.loaders.external import DATASET_SPECS


def test_source_quotas_total_1500():
    assert sum(SOURCE_QUOTAS.values()) == 1500
    assert set(SOURCE_QUOTAS) == set(DATASET_SPECS) - {"temprageval"}


def test_builder_selects_without_replacement_and_writes_manifest(tmp_path: Path):
    cache = tmp_path / "cache"
    for dataset_id, quota in SOURCE_QUOTAS.items():
        spec = DATASET_SPECS[dataset_id]
        path = cache / dataset_id / f"{spec.default_split or 'default'}.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "".join(json.dumps({"id": f"{dataset_id}-{index}"}) + "\n" for index in range(quota + 2)),
            encoding="utf-8",
        )
    output = tmp_path / "source.jsonl"
    manifest = build(cache, output, seed=7)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1500
    assert len({row["river_source_id"] for row in rows}) == 1500
    assert manifest["scientific_finalization_eligible"] is False
    assert output.with_suffix(".jsonl.manifest.json").exists()
    with pytest.raises(FileExistsError):
        build(cache, output, seed=7)
