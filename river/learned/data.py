"""Batch construction for RIVER-L."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from river.schemas import RevisionState

from .features import RiverLFeatureExtractor, max_observed_time, target_time_from_example


STATE_TO_INDEX = {
    RevisionState.ACTIVE.value: 0,
    RevisionState.DISPUTED.value: 1,
    RevisionState.EVOLUTION.value: 2,
    RevisionState.UNKNOWN.value: 3,
    RevisionState.ABSTAIN.value: 4,
}
INDEX_TO_STATE = {value: key for key, value in STATE_TO_INDEX.items()}


@dataclass
class RiverLBatch:
    evidence_features: torch.Tensor
    evidence_mask: torch.Tensor
    hypothesis_mask: torch.Tensor
    state_targets: torch.Tensor
    selected_targets: torch.Tensor
    update_mask: torch.Tensor
    example_ids: list[str]
    hypothesis_ids: list[list[str]]
    pair_ids: list[str]

    def to(self, device: torch.device | str) -> "RiverLBatch":
        return RiverLBatch(
            evidence_features=self.evidence_features.to(device),
            evidence_mask=self.evidence_mask.to(device),
            hypothesis_mask=self.hypothesis_mask.to(device),
            state_targets=self.state_targets.to(device),
            selected_targets=self.selected_targets.to(device),
            update_mask=self.update_mask.to(device),
            example_ids=self.example_ids,
            hypothesis_ids=self.hypothesis_ids,
            pair_ids=self.pair_ids,
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_hash(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def collate_examples(
    examples: list[dict[str, Any]],
    extractor: RiverLFeatureExtractor,
    *,
    device: torch.device | str = "cpu",
) -> RiverLBatch:
    batch_size = len(examples)
    max_hypotheses = max((len(item.get("hypotheses", [])) for item in examples), default=1)
    max_evidence_per_hypothesis = 1
    grouped_features: list[list[list[list[float]]]] = []
    hypothesis_ids: list[list[str]] = []

    for example in examples:
        evidence = list(example.get("evidence", []))
        by_id = {item["evidence_id"]: item for item in evidence if "evidence_id" in item}
        group_counts = Counter(
            item.get("independence_group", item.get("evidence_id", "")) for item in evidence
        )
        target_time = target_time_from_example(example)
        observed_max = max_observed_time(evidence)
        per_hypothesis: list[list[list[float]]] = []
        ids: list[str] = []
        for hypothesis in example.get("hypotheses", []):
            ids.append(hypothesis["hypothesis_id"])
            rows = [
                extractor.evidence_features(
                    by_id[evidence_id],
                    group_counts=group_counts,
                    target_time=target_time,
                    max_observed_at=observed_max,
                )
                for evidence_id in hypothesis.get("evidence_ids", [])
                if evidence_id in by_id
            ]
            per_hypothesis.append(rows)
            max_evidence_per_hypothesis = max(max_evidence_per_hypothesis, len(rows))
        grouped_features.append(per_hypothesis)
        hypothesis_ids.append(ids)

    feature_dim = extractor.feature_dim
    features = torch.zeros(batch_size, max_hypotheses, max_evidence_per_hypothesis, feature_dim, dtype=torch.float32)
    evidence_mask = torch.zeros(batch_size, max_hypotheses, max_evidence_per_hypothesis, dtype=torch.float32)
    hypothesis_mask = torch.zeros(batch_size, max_hypotheses, dtype=torch.float32)
    selected_targets = torch.full((batch_size,), -100, dtype=torch.long)
    state_targets = torch.zeros(batch_size, dtype=torch.long)
    update_mask = torch.zeros(batch_size, dtype=torch.bool)

    for batch_index, example in enumerate(examples):
        state_targets[batch_index] = STATE_TO_INDEX[str(example.get("gold_state", RevisionState.UNKNOWN.value))]
        ids = hypothesis_ids[batch_index]
        hypothesis_mask[batch_index, : len(ids)] = 1.0
        gold_hypothesis_id = example.get("gold_hypothesis_id")
        if gold_hypothesis_id in ids:
            selected_targets[batch_index] = ids.index(gold_hypothesis_id)
        update_mask[batch_index] = (
            example.get("intervention") in {"supported_genuine_update", "supported_update"}
            and gold_hypothesis_id is not None
            and gold_hypothesis_id == example.get("intervention_hypothesis_id")
        )
        for h_index, rows in enumerate(grouped_features[batch_index]):
            for e_index, row in enumerate(rows):
                features[batch_index, h_index, e_index, :] = torch.tensor(row, dtype=torch.float32)
                evidence_mask[batch_index, h_index, e_index] = 1.0

    return RiverLBatch(
        evidence_features=features.to(device),
        evidence_mask=evidence_mask.to(device),
        hypothesis_mask=hypothesis_mask.to(device),
        state_targets=state_targets.to(device),
        selected_targets=selected_targets.to(device),
        update_mask=update_mask.to(device),
        example_ids=[item.get("example_id", f"example-{i}") for i, item in enumerate(examples)],
        hypothesis_ids=hypothesis_ids,
        pair_ids=[item.get("pair_id", "") for item in examples],
    )


def permute_evidence(batch: RiverLBatch, *, seed: int) -> RiverLBatch:
    generator = torch.Generator(device=batch.evidence_features.device)
    generator.manual_seed(seed)
    features = batch.evidence_features.clone()
    masks = batch.evidence_mask.clone()
    for b_index in range(features.shape[0]):
        for h_index in range(features.shape[1]):
            valid = int(masks[b_index, h_index].sum().item())
            if valid <= 1:
                continue
            order = torch.randperm(valid, generator=generator, device=features.device)
            features[b_index, h_index, :valid] = features[b_index, h_index, :valid][order]
            masks[b_index, h_index, :valid] = masks[b_index, h_index, :valid][order]
    return RiverLBatch(
        evidence_features=features,
        evidence_mask=masks,
        hypothesis_mask=batch.hypothesis_mask,
        state_targets=batch.state_targets,
        selected_targets=batch.selected_targets,
        update_mask=batch.update_mask,
        example_ids=batch.example_ids,
        hypothesis_ids=batch.hypothesis_ids,
        pair_ids=batch.pair_ids,
    )
