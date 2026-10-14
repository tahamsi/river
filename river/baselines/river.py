from __future__ import annotations

from datetime import datetime
from pathlib import Path

from river.revision import RIVERConfig, RIVERDeterministic
from river.schemas import BeliefHypothesis, EvidenceRecord, RevisionState

from .base import Baseline, BaselineCapabilities, BaselineUnavailable


def _decision_confidence(decision) -> float | None:
    """The selected hypothesis's share of the total decision score.

    A monotone transform of the scores the reviser already computed, so it
    reports the decision rather than dressing it up. Negative scores are clamped
    because a share is only meaningful over non-negative mass.

    None when nothing was selected or all mass is zero: there is no decision to
    be confident about, and defaulting to a number would put a fabricated
    confidence into the calibration table.
    """

    scores = getattr(decision, "scores", None) or {}
    selected = getattr(decision, "selected_hypothesis_id", None)
    if selected is None or selected not in scores:
        return None
    total = sum(max(float(value), 0.0) for value in scores.values())
    if total <= 0.0:
        return None
    return min(1.0, max(0.0, float(scores[selected]) / total))


class RIVERDBaseline(Baseline):
    name = "river_d"
    version = "0.2.0"
    capabilities = BaselineCapabilities(True, True, True, True, True, False)

    def __init__(self, config=None):
        super().__init__(config)
        river_keys = RIVERConfig.__dataclass_fields__.keys()
        self.reviser = RIVERDeterministic(RIVERConfig(**{k: v for k, v in self.config.items() if k in river_keys}))

    def process_turn(self, session_id, query, retrieved_chunks, target_time=None):
        hypotheses = [
            item if isinstance(item, BeliefHypothesis) else BeliefHypothesis.model_validate(item)
            for item in self.config.get("hypotheses", [])
        ]
        evidence = [
            item if isinstance(item, EvidenceRecord) else EvidenceRecord.model_validate(item)
            for item in retrieved_chunks
        ]
        if isinstance(target_time, str):
            target_time = datetime.fromisoformat(target_time.replace("Z", "+00:00"))
        decision = self.reviser.resolve(hypotheses, evidence, target_time)
        return {
            "answer": decision.selected_hypothesis_id or "",
            "predicted_state": decision.state.value,
            "predicted_hypothesis_id": decision.selected_hypothesis_id,
            "confidence": _decision_confidence(decision),
            "revision": decision.model_dump(mode="json"),
        }


class RIVERLBaseline(Baseline):
    name = "river_l"
    version = "river-l-deepsets-0.1.0"
    capabilities = BaselineCapabilities(True, True, True, True, True, False)

    def __init__(self, config=None):
        super().__init__(config)
        checkpoint = self.config.get("checkpoint") or self.config.get("checkpoint_path")
        if not checkpoint:
            raise BaselineUnavailable(
                "RIVER-L requires a trained checkpoint. Train and freeze it before evaluation; "
                "RIVER-D remains fully executable without learned weights."
            )
        try:
            from river.learned.checkpoint import load_checkpoint
        except ModuleNotFoundError as exc:
            raise BaselineUnavailable("RIVER-L requires PyTorch. Install with uv sync --extra learned.") from exc
        try:
            self.model, self.extractor, self.checkpoint_manifest = load_checkpoint(
                Path(checkpoint),
                map_location=self.config.get("device", "cpu"),
            )
        except Exception as exc:
            raise BaselineUnavailable(f"Could not load RIVER-L checkpoint {checkpoint!r}: {exc}") from exc

    def process_turn(self, session_id, query, retrieved_chunks, target_time=None):
        del session_id, query
        try:
            import torch
            from river.learned.data import INDEX_TO_STATE, collate_examples
        except ModuleNotFoundError as exc:
            raise BaselineUnavailable("RIVER-L requires PyTorch. Install with uv sync --extra learned.") from exc

        hypotheses = [
            item.model_dump(mode="json") if isinstance(item, BeliefHypothesis) else item
            for item in self.config.get("hypotheses", [])
        ]
        evidence = [
            item.model_dump(mode="json") if isinstance(item, EvidenceRecord) else item
            for item in retrieved_chunks
        ]
        example = {
            "example_id": "inference",
            "pair_id": "",
            "gold_state": RevisionState.UNKNOWN.value,
            "target_time": target_time,
            "evidence": evidence,
            "hypotheses": hypotheses,
        }
        batch = collate_examples([example], self.extractor)
        with torch.no_grad():
            outputs = self.model(batch.evidence_features, batch.evidence_mask, batch.hypothesis_mask)
            state_probs = torch.softmax(outputs["state_logits"], dim=-1)[0]
            hypothesis_probs = torch.softmax(outputs["hypothesis_logits"], dim=-1)[0]

        state_index = int(state_probs.argmax().item())
        state = INDEX_TO_STATE[state_index]
        hypothesis_ids = batch.hypothesis_ids[0]
        selected_index = int(hypothesis_probs.argmax().item()) if hypothesis_ids else -1
        selected_hypothesis_id = (
            hypothesis_ids[selected_index]
            if state == RevisionState.ACTIVE.value and 0 <= selected_index < len(hypothesis_ids)
            else None
        )
        return {
            "answer": selected_hypothesis_id or "",
            "predicted_state": state,
            "predicted_hypothesis_id": selected_hypothesis_id,
            "confidence": float(outputs["confidence"][0].item()),
            "state_probabilities": {
                INDEX_TO_STATE[index]: float(value.item())
                for index, value in enumerate(state_probs)
            },
            "revision": {
                "selected_hypothesis_distribution": {
                    hypothesis_ids[index]: float(hypothesis_probs[index].item())
                    for index in range(len(hypothesis_ids))
                },
                "checkpoint_manifest": self.checkpoint_manifest,
                "feature_names": self.extractor.feature_names,
            },
        }
