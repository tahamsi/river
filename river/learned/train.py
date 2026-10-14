"""Train RIVER-L checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch

from .checkpoint import save_checkpoint
from .data import RiverLBatch, collate_examples, dataset_hash, load_jsonl, permute_evidence
from .features import PROHIBITED_TRUTH_FEATURES, RiverLFeatureConfig, RiverLFeatureExtractor
from .losses import RiverLLossConfig, total_river_l_loss
from .model import DeepSetsRIVERL, RiverLModelConfig


def stable_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _batches(examples: list[dict[str, Any]], batch_size: int, seed: int):
    order = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start:start + batch_size]]


def _evaluate(model: DeepSetsRIVERL, batch: RiverLBatch, loss_config: RiverLLossConfig) -> dict[str, float]:
    with torch.no_grad():
        outputs = model(batch.evidence_features, batch.evidence_mask, batch.hypothesis_mask)
        permuted = model(*_batch_args(permute_evidence(batch, seed=0)))
        loss, parts = total_river_l_loss(outputs, permuted, batch, loss_config)
        state_acc = outputs["state_logits"].argmax(dim=-1).eq(batch.state_targets).float().mean()
        selected_mask = batch.selected_targets >= 0
        if selected_mask.any():
            selected_acc = outputs["hypothesis_logits"][selected_mask].argmax(dim=-1).eq(
                batch.selected_targets[selected_mask]
            ).float().mean()
        else:
            selected_acc = torch.tensor(0.0)
    parts.update({
        "eval_loss": float(loss.detach().cpu()),
        "state_accuracy": float(state_acc.detach().cpu()),
        "selected_accuracy": float(selected_acc.detach().cpu()),
    })
    return parts


def _batch_args(batch: RiverLBatch):
    return batch.evidence_features, batch.evidence_mask, batch.hypothesis_mask


def train_from_config(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_seed(seed)

    device = getattr(args, "device", None) or config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    feature_config = RiverLFeatureConfig(**config.get("features", {}))
    extractor = RiverLFeatureExtractor(feature_config)
    if not feature_config.include_arrival_feature:
        extractor.validate_standard_features()

    train_examples = load_jsonl(args.train)
    validation_examples = load_jsonl(args.validation) if args.validation else train_examples
    validation_batch = collate_examples(validation_examples, extractor, device=device)

    model_config = RiverLModelConfig(
        feature_dim=extractor.feature_dim,
        **config.get("model", {}),
    )
    model = DeepSetsRIVERL(model_config).to(device)
    loss_config = RiverLLossConfig(**config.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 0.01)))
    epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 50))
    batch_size = int(config.get("batch_size", len(train_examples) or 1))
    patience = int(config.get("early_stopping_patience", 8))

    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        for chunk in _batches(train_examples, batch_size, seed + epoch):
            batch = collate_examples(chunk, extractor, device=device)
            outputs = model(*_batch_args(batch))
            permuted_batch = permute_evidence(batch, seed=seed + epoch)
            permuted_outputs = model(*_batch_args(permuted_batch))
            loss, _parts = total_river_l_loss(outputs, permuted_outputs, batch, loss_config)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        metrics = _evaluate(model, validation_batch, loss_config)
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        if metrics["eval_loss"] < best_loss - 1e-8:
            best_loss = metrics["eval_loss"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")

    manifest = {
        "model_family": "river_l_deepsets",
        "architecture": "Deep Sets",
        "engineering_validation_only": bool(config.get("engineering_validation_only", False)),
        "seed": seed,
        "config_hash": stable_hash(config),
        "train_dataset_hash": dataset_hash(args.train),
        "validation_dataset_hash": dataset_hash(args.validation) if args.validation else dataset_hash(args.train),
        "feature_names": extractor.feature_names,
        "prohibited_standard_features": sorted(PROHIBITED_TRUTH_FEATURES),
        "best_validation_loss": best_loss,
        "epochs_ran": len(history),
        "history": history,
    }
    save_checkpoint(args.output, model, extractor, manifest=manifest)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if args.model_card:
        write_model_card(args.model_card, manifest)
    return manifest


def write_model_card(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# RIVER-L Checkpoint

- Model family: `{manifest['model_family']}`
- Architecture: `{manifest['architecture']}`
- Seed: `{manifest['seed']}`
- Config hash: `{manifest['config_hash']}`
- Train dataset hash: `{manifest['train_dataset_hash']}`
- Validation dataset hash: `{manifest['validation_dataset_hash']}`
- Best validation loss: `{manifest['best_validation_loss']}`
- Epochs run: `{manifest['epochs_ran']}`

This card records an engineering checkpoint manifest. Do not claim scientific
performance improvements from this checkpoint without running labelled
evaluation experiments.
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a learned RIVER-L checkpoint")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-card", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    return parser


def main() -> None:
    manifest = train_from_config(build_parser().parse_args())
    print(json.dumps({k: manifest[k] for k in ("config_hash", "best_validation_loss", "epochs_ran")}, indent=2))


if __name__ == "__main__":
    main()
