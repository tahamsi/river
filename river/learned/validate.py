"""Validate frozen RIVER-L checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .data import collate_examples, dataset_hash, load_jsonl


def validate_checkpoint(checkpoint: Path, dataset: Path) -> dict:
    model, extractor, manifest = load_checkpoint(checkpoint)
    examples = load_jsonl(dataset)
    batch = collate_examples(examples, extractor)
    with torch.no_grad():
        outputs = model(batch.evidence_features, batch.evidence_mask, batch.hypothesis_mask)
        state_pred = outputs["state_logits"].argmax(dim=-1)
        state_accuracy = state_pred.eq(batch.state_targets).float().mean().item()
        selected_mask = batch.selected_targets >= 0
        selected_accuracy = (
            outputs["hypothesis_logits"][selected_mask].argmax(dim=-1).eq(batch.selected_targets[selected_mask]).float().mean().item()
            if selected_mask.any()
            else 0.0
        )
    return {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "dataset_hash": dataset_hash(dataset),
        "state_accuracy": state_accuracy,
        "selected_accuracy": selected_accuracy,
        "example_count": len(examples),
        "manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a frozen RIVER-L checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = validate_checkpoint(args.checkpoint, args.fixture)
    text = json.dumps(metrics, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
