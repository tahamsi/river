"""Command line tools for building and validating RIVER-Bench fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loaders import DATASET_SPECS, download_external_dataset
from .transformations import build_authored_examples, load_seed_manifest
from .validation import (
    examples_to_jsonl,
    intervention_counts,
    load_examples,
    validate_all,
    validate_order_only_equivalence,
    validate_pair_completeness,
    validate_split_leakage,
    validate_unique_ids,
)


DEFAULT_SEED_MANIFEST = Path("river_bench/manifests/river_bench_seed_fixtures.jsonl")
DEFAULT_UNIT_OUTPUT = Path("river_bench/processed/river_bench_unit.jsonl")


def _load_fixture(args: argparse.Namespace):
    return load_examples(args.fixture)


def cmd_build_authored(args: argparse.Namespace) -> None:
    seeds = load_seed_manifest(args.seed_manifest)
    examples = build_authored_examples(seeds)
    validate_all(examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(examples_to_jsonl(examples), encoding="utf-8")
    print(f"wrote {len(examples)} examples to {args.output}")
    print(json.dumps(intervention_counts(examples), indent=2, sort_keys=True))


def cmd_validate_schema(args: argparse.Namespace) -> None:
    examples = _load_fixture(args)
    validate_unique_ids(examples)
    print(f"schema ok: {len(examples)} examples")


def cmd_validate_pairs(args: argparse.Namespace) -> None:
    examples = _load_fixture(args)
    validate_pair_completeness(examples)
    print("pair completeness ok")


def cmd_validate_order(args: argparse.Namespace) -> None:
    examples = _load_fixture(args)
    validate_order_only_equivalence(examples)
    print("order-only equivalence ok")


def cmd_validate_leakage(args: argparse.Namespace) -> None:
    examples = _load_fixture(args)
    validate_split_leakage(examples)
    print("split leakage ok")


def cmd_count(args: argparse.Namespace) -> None:
    examples = _load_fixture(args)
    print(json.dumps(intervention_counts(examples), indent=2, sort_keys=True))


def cmd_validate_all(args: argparse.Namespace) -> None:
    examples = _load_fixture(args)
    validate_all(examples)
    print(f"all validations ok: {len(examples)} examples")


def cmd_download_external(args: argparse.Namespace) -> None:
    path = download_external_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        split=args.split,
        limit=args.limit,
        manual_file=args.manual_file,
    )
    print(f"cached {args.dataset} at {path}")


def cmd_list_external(args: argparse.Namespace) -> None:
    del args
    rows = {
        dataset_id: {
            "status": spec.status,
            "hf_dataset": spec.hf_dataset,
            "hf_config": spec.hf_config,
            "default_split": spec.default_split,
            "notes": spec.notes,
        }
        for dataset_id, spec in sorted(DATASET_SPECS.items())
    }
    print(json.dumps(rows, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate RIVER-Bench fixtures")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-authored", help="Build deterministic examples from local seed fixtures")
    build.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    build.add_argument("--output", type=Path, default=DEFAULT_UNIT_OUTPUT)
    build.set_defaults(func=cmd_build_authored)

    for name, help_text, func in [
        ("validate-schema", "Validate JSONL records against the schema and unique IDs", cmd_validate_schema),
        ("validate-pairs", "Validate pair completeness and same-split pairing", cmd_validate_pairs),
        ("validate-order-equivalence", "Validate order reversal changes only evidence order", cmd_validate_order),
        ("validate-split-leakage", "Validate base/entity/event leakage across splits", cmd_validate_leakage),
        ("count-interventions", "Display intervention counts", cmd_count),
        ("validate-all", "Run all local validations", cmd_validate_all),
    ]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--fixture", type=Path, default=DEFAULT_UNIT_OUTPUT)
        command.set_defaults(func=func)

    download = sub.add_parser("download-external", help="Explicitly download/cache an external source dataset")
    download.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    download.add_argument("--cache-dir", type=Path, default=Path("data"))
    download.add_argument("--split")
    download.add_argument("--limit", type=int)
    download.add_argument("--manual-file", type=Path)
    download.set_defaults(func=cmd_download_external)

    list_external = sub.add_parser("list-external", help="Show external loader hook status")
    list_external.set_defaults(func=cmd_list_external)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
