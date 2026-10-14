"""Permutation-invariant RIVER-L model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class RiverLModelConfig:
    feature_dim: int
    hidden_dim: int = 32
    state_count: int = 5
    dropout: float = 0.0

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


class DeepSetsRIVERL(nn.Module):
    """Deep Sets evidence revision model.

    Evidence positions are never embedded. Per-evidence features are encoded by
    ``phi`` and aggregated by masked mean before hypothesis/state prediction.
    """

    def __init__(self, config: RiverLModelConfig):
        super().__init__()
        self.config = config
        self.phi = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.hypothesis_head = nn.Linear(config.hidden_dim, 1)
        self.state_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.state_count),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        evidence_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        hypothesis_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.phi(evidence_features)
        mask = evidence_mask.unsqueeze(-1)
        evidence_counts = evidence_mask.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        hypothesis_repr = (encoded * mask).sum(dim=2) / evidence_counts

        hypothesis_logits = self.hypothesis_head(hypothesis_repr).squeeze(-1)
        hypothesis_logits = hypothesis_logits.masked_fill(hypothesis_mask == 0, -1e9)

        hmask = hypothesis_mask.unsqueeze(-1)
        hypothesis_count = hypothesis_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        global_mean = (hypothesis_repr * hmask).sum(dim=1) / hypothesis_count
        global_max = hypothesis_repr.masked_fill(hmask == 0, -1e9).max(dim=1).values
        global_max = torch.where(torch.isfinite(global_max), global_max, torch.zeros_like(global_max))
        completeness = (evidence_mask.sum(dim=(1, 2), keepdim=False) > 0).float().unsqueeze(-1)
        density = evidence_mask.sum(dim=(1, 2), keepdim=False).unsqueeze(-1) / (
            hypothesis_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        global_repr = torch.cat([global_mean, global_max, completeness, density], dim=-1)
        state_logits = self.state_head(global_repr)
        confidence = self.confidence_head(global_repr).squeeze(-1)
        return {
            "hypothesis_logits": hypothesis_logits,
            "state_logits": state_logits,
            "confidence": confidence,
        }
