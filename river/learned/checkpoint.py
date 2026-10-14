"""Checkpoint save/load for RIVER-L."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .features import RiverLFeatureConfig, RiverLFeatureExtractor
from .model import DeepSetsRIVERL, RiverLModelConfig


CHECKPOINT_TYPE = "river_river_l_deepsets_checkpoint"
CHECKPOINT_VERSION = "0.1.0"


def save_checkpoint(
    path: Path,
    model: DeepSetsRIVERL,
    extractor: RiverLFeatureExtractor,
    *,
    manifest: dict[str, Any],
) -> None:
    payload = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_config": model.config.manifest(),
        "feature_config": extractor.config.manifest(),
        "feature_names": extractor.feature_names,
        "state_dict": model.state_dict(),
        "manifest": manifest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> tuple[DeepSetsRIVERL, RiverLFeatureExtractor, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(f"{path} is not a valid RIVER-L checkpoint")
    model_config = RiverLModelConfig(**payload["model_config"])
    feature_config = RiverLFeatureConfig(**payload["feature_config"])
    extractor = RiverLFeatureExtractor(feature_config)
    if payload.get("feature_names") != extractor.feature_names:
        raise ValueError("checkpoint feature names do not match feature configuration")
    model = DeepSetsRIVERL(model_config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, extractor, payload.get("manifest", {})
