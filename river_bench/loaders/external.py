"""Explicit hooks for external RIVER-Bench source datasets.

These hooks never run from tests and never download implicitly. They either
write raw JSONL into a caller-provided cache directory or report that the
dataset requires manual download/credentials.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExternalDatasetSpec:
    dataset_id: str
    status: str
    notes: str
    hf_dataset: str | None = None
    hf_config: str | None = None
    default_split: str | None = None
    hf_data_file: str | None = None


DATASET_SPECS: dict[str, ExternalDatasetSpec] = {
    "fever": ExternalDatasetSpec(
        dataset_id="fever",
        status="implemented_explicit_hf",
        hf_dataset="fever/fever",
        hf_config="v1.0",
        default_split="paper_dev",
        # FEVER's parquet conversion is pinned because it is not present on
        # the repository's ``main`` branch.
        hf_data_file=(
            "https://huggingface.co/datasets/fever/fever/resolve/"
            "1c55aad95efe327fc6e71a5893d6040d93a8f98c/v1.0/fever-paper_dev.parquet"
        ),
        notes="Uses the FEVER source labels, including NOT ENOUGH INFO, when HuggingFace access is available.",
    ),
    "temprageval": ExternalDatasetSpec(
        dataset_id="temprageval",
        status="implemented_explicit_hf",
        hf_dataset="siyue/TempRAGEval",
        default_split="test",
        notes="HuggingFace access is gated; accept the dataset terms and set HF_TOKEN before downloading.",
    ),
    "pat_questions": ExternalDatasetSpec(
        dataset_id="pat_questions",
        status="manual_or_credentials_required",
        notes="Public official release: github.com/jannatmeem95/PAT-Questions. Export the selected dated snapshot to JSONL with --manual-file.",
    ),
    "wikicontradict": ExternalDatasetSpec(
        dataset_id="wikicontradict",
        status="implemented_explicit_hf",
        hf_dataset="ibm-research/Wikipedia_contradict_benchmark",
        default_split="train",
        notes="Official IBM WikiContradict release on HuggingFace.",
    ),
    "longmemeval": ExternalDatasetSpec(
        dataset_id="longmemeval",
        status="implemented_explicit_hf",
        hf_dataset="xiaowu0162/longmemeval-cleaned",
        default_split="train",
        notes="Official cleaned LongMemEval release; use the oracle file/config when preparing exact evidence.",
    ),
    "2wikimultihopqa": ExternalDatasetSpec(
        dataset_id="2wikimultihopqa",
        status="implemented_explicit_hf",
        hf_dataset="xanhho/2WikiMultihopQA",
        default_split="validation",
        notes="Official 2WikiMultihopQA Hugging Face mirror maintained by the dataset author. This hook does not substitute HotpotQA.",
    ),
}


def _copy_manual_file(dataset_id: str, manual_file: Path, cache_dir: Path) -> Path:
    if not manual_file.exists():
        raise FileNotFoundError(manual_file)
    out_path = cache_dir / dataset_id / "manual.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with manual_file.open(encoding="utf-8") as source, out_path.open("w", encoding="utf-8") as target:
        for line in source:
            if line.strip():
                target.write(line)
    return out_path


def _download_hf_jsonl(spec: ExternalDatasetSpec, cache_dir: Path, split: str | None, limit: int | None) -> Path:
    if spec.hf_dataset is None:
        raise ValueError(f"{spec.dataset_id} does not define a HuggingFace dataset hook")

    out_path = cache_dir / spec.dataset_id / f"{split or spec.default_split or 'default'}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    code = """
import json
from pathlib import Path

import datasets as hf_datasets

dataset_name = {dataset_name!r}
config_name = {config_name!r}
split = {split!r}
limit = {limit!r}
out_path = Path({out_path!r})
data_file = {data_file!r}

if data_file is not None:
    ds = hf_datasets.load_dataset("parquet", data_files={{split: data_file}}, split=split)
elif config_name is None:
    ds = hf_datasets.load_dataset(dataset_name, split=split)
else:
    ds = hf_datasets.load_dataset(dataset_name, config_name, split=split)

if limit is not None:
    ds = ds.select(range(min(limit, len(ds))))

with out_path.open("w", encoding="utf-8") as handle:
    for row in ds:
        payload = dict(row)
        if dataset_name in {{"fever", "fever/fever"}}:
            label = str(payload.get("label", "")).upper().replace(" ", "_")
            payload["_river_bench_label_is_nei"] = label in {{"NOT_ENOUGH_INFO", "NEI"}}
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\\n")
print(out_path)
""".format(
        dataset_name=spec.hf_dataset,
        config_name=spec.hf_config,
        split=split or spec.default_split,
        limit=limit,
        out_path=str(out_path),
        data_file=spec.hf_data_file,
    )
    # ``river_bench`` deliberately avoids shadowing HuggingFace's ``datasets``
    # package, so the subprocess can run from the project working directory.
    subprocess.run([sys.executable, "-c", code], check=True)
    return out_path


def download_external_dataset(
    dataset_id: str,
    *,
    cache_dir: Path,
    split: str | None = None,
    limit: int | None = None,
    manual_file: Path | None = None,
) -> Path:
    try:
        spec = DATASET_SPECS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"unknown external dataset hook {dataset_id!r}") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    if manual_file is not None:
        return _copy_manual_file(dataset_id, manual_file, cache_dir)
    if spec.status == "implemented_explicit_hf":
        return _download_hf_jsonl(spec, cache_dir, split, limit)
    raise NotImplementedError(
        f"{dataset_id} requires manual download or credentials. {spec.notes} "
        "Pass --manual-file to cache an already downloaded JSONL export."
    )


def load_external_records(dataset_id: str, *, cache_dir: Path, split: str | None = None) -> list[dict[str, Any]]:
    spec = DATASET_SPECS.get(dataset_id)
    if spec is None:
        raise ValueError(f"unknown external dataset hook {dataset_id!r}")
    candidates = [
        cache_dir / dataset_id / f"{split or spec.default_split or 'default'}.jsonl",
        cache_dir / dataset_id / "manual.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise FileNotFoundError(
        f"No cached {dataset_id} export found under {cache_dir}. "
        "Run python -m river_bench.river_bench download-external explicitly first."
    )
