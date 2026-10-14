"""RIVER-L objective terms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .data import RiverLBatch


@dataclass(frozen=True)
class RiverLLossConfig:
    lambda_inv: float = 1.0
    lambda_upd: float = 1.0
    lambda_cal: float = 0.1

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def revision_loss(outputs: dict[str, torch.Tensor], batch: RiverLBatch) -> torch.Tensor:
    state = F.cross_entropy(outputs["state_logits"], batch.state_targets)
    selected_mask = batch.selected_targets >= 0
    if selected_mask.any():
        selected = F.cross_entropy(
            outputs["hypothesis_logits"][selected_mask],
            batch.selected_targets[selected_mask],
        )
    else:
        selected = outputs["state_logits"].sum() * 0.0
    return state + selected


def invariance_loss(
    original_outputs: dict[str, torch.Tensor],
    permuted_outputs: dict[str, torch.Tensor],
    batch: RiverLBatch,
) -> torch.Tensor:
    state = F.mse_loss(original_outputs["state_logits"], permuted_outputs["state_logits"])
    mask = batch.hypothesis_mask.bool()
    if mask.any():
        hyp = F.mse_loss(
            original_outputs["hypothesis_logits"][mask],
            permuted_outputs["hypothesis_logits"][mask],
        )
    else:
        hyp = state * 0.0
    conf = F.mse_loss(original_outputs["confidence"], permuted_outputs["confidence"])
    return state + hyp + conf


def update_loss(outputs: dict[str, torch.Tensor], batch: RiverLBatch) -> torch.Tensor:
    mask = batch.update_mask & (batch.selected_targets >= 0)
    if not mask.any():
        return outputs["state_logits"].sum() * 0.0
    selected = F.cross_entropy(outputs["hypothesis_logits"][mask], batch.selected_targets[mask])
    active_targets = torch.zeros(mask.sum(), dtype=torch.long, device=batch.state_targets.device)
    active = F.cross_entropy(outputs["state_logits"][mask], active_targets)
    return selected + active


def calibration_loss(outputs: dict[str, torch.Tensor], batch: RiverLBatch) -> torch.Tensor:
    with torch.no_grad():
        state_correct = outputs["state_logits"].argmax(dim=-1).eq(batch.state_targets)
        selected_mask = batch.selected_targets >= 0
        selected_correct = torch.ones_like(state_correct)
        selected_correct[selected_mask] = outputs["hypothesis_logits"][selected_mask].argmax(dim=-1).eq(
            batch.selected_targets[selected_mask]
        )
        target = (state_correct & selected_correct).float()
    return F.mse_loss(outputs["confidence"], target)


def total_river_l_loss(
    outputs: dict[str, torch.Tensor],
    permuted_outputs: dict[str, torch.Tensor],
    batch: RiverLBatch,
    config: RiverLLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_revision = revision_loss(outputs, batch)
    l_inv = invariance_loss(outputs, permuted_outputs, batch)
    l_update = update_loss(outputs, batch)
    l_cal = calibration_loss(outputs, batch)
    total = l_revision + config.lambda_inv * l_inv + config.lambda_upd * l_update + config.lambda_cal * l_cal
    return total, {
        "loss": float(total.detach().cpu()),
        "revision_loss": float(l_revision.detach().cpu()),
        "invariance_loss": float(l_inv.detach().cpu()),
        "update_loss": float(l_update.detach().cpu()),
        "calibration_loss": float(l_cal.detach().cpu()),
    }
