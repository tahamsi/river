"""RIVER: recency-invariant, validity-sensitive evidence revision."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from .schemas import (
    BeliefHypothesis,
    EvidenceRecord,
    RevisionDecision,
    RevisionState,
)


@dataclass(frozen=True)
class RIVERConfig:
    support_threshold: float = 0.60
    margin_threshold: float = 0.15
    abstention_threshold: float = 0.30
    contradiction_penalty: float = 1.0
    use_valid_time: bool = True
    use_independence_discount: bool = True
    use_contradiction: bool = True
    use_provenance: bool = True
    allow_abstention: bool = True
    use_decision_margin: bool = True
    use_arrival_as_truth_feature: bool = False

    def __post_init__(self) -> None:
        if self.use_arrival_as_truth_feature:
            # Available only as the explicit RIVER+Arrival negative control.
            return
        for name in ("support_threshold", "margin_threshold", "abstention_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


class RIVERDeterministic:
    """Order-invariant revision over a multiset of assessed evidence.

    ``observed_at`` is excluded from the default truth score. Evidence order is
    used only by the explicit RIVER+Arrival negative-control configuration.
    """

    def __init__(self, config: RIVERConfig | None = None):
        self.config = config or RIVERConfig()

    def _scores(
        self,
        hypotheses: list[BeliefHypothesis],
        evidence: list[EvidenceRecord],
        target_time: datetime | None,
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        by_id = {item.evidence_id: item for item in evidence}
        group_sizes = Counter(item.independence_group for item in evidence)
        scores: dict[str, float] = {}
        applicable: dict[str, list[str]] = {}

        max_observed = max((item.observed_at for item in evidence), default=None)
        for hypothesis in hypotheses:
            score = 0.0
            used: list[str] = []
            for evidence_id in sorted(hypothesis.evidence_ids):
                item = by_id.get(evidence_id)
                if item is None:
                    continue
                temporal = item.valid_time.contains(target_time) if self.config.use_valid_time else 1.0
                if temporal == 0.0:
                    continue
                dependence = (
                    1.0 / math.sqrt(group_sizes[item.independence_group])
                    if self.config.use_independence_discount
                    else 1.0
                )
                provenance = item.provenance_reliability if self.config.use_provenance else 1.0
                contradiction = item.contradiction if self.config.use_contradiction else 0.0
                signed = item.entailment - self.config.contradiction_penalty * contradiction
                contribution = temporal * provenance * item.extraction_confidence * dependence * signed
                if self.config.use_arrival_as_truth_feature and max_observed is not None:
                    age_days = max((max_observed - item.observed_at).total_seconds() / 86400.0, 0.0)
                    contribution *= math.exp(-0.01 * age_days)
                score += contribution
                used.append(evidence_id)
            scores[hypothesis.hypothesis_id] = score
            applicable[hypothesis.hypothesis_id] = used
        return scores, applicable

    def resolve(
        self,
        hypotheses: list[BeliefHypothesis],
        evidence: list[EvidenceRecord],
        target_time: datetime | None = None,
    ) -> RevisionDecision:
        if not hypotheses or not evidence:
            state = RevisionState.ABSTAIN if self.config.allow_abstention else RevisionState.UNKNOWN
            return RevisionDecision(state=state, target_time=target_time, reason_codes=["no_evidence"])

        scores, applicable = self._scores(hypotheses, evidence, target_time)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_id, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
        margin = top_score - second_score if len(ranked) > 1 else top_score

        if top_score < self.config.abstention_threshold and self.config.allow_abstention:
            return RevisionDecision(
                state=RevisionState.ABSTAIN,
                top_score=top_score,
                score_margin=margin,
                target_time=target_time,
                scores=scores,
                reason_codes=["insufficient_evidence"],
            )

        supported = [item for item in ranked if item[1] >= self.config.support_threshold]
        if len(supported) >= 2:
            hypothesis_by_id = {item.hypothesis_id: item for item in hypotheses}
            evidence_by_id = {item.evidence_id: item for item in evidence}
            first, second = hypothesis_by_id[supported[0][0]], hypothesis_by_id[supported[1][0]]
            first_intervals = [evidence_by_id[e].valid_time for e in applicable[first.hypothesis_id]]
            second_intervals = [evidence_by_id[e].valid_time for e in applicable[second.hypothesis_id]]
            if first_intervals and second_intervals and all(
                not left.overlaps(right) for left in first_intervals for right in second_intervals
            ):
                return RevisionDecision(
                    state=RevisionState.EVOLUTION,
                    alternative_hypothesis_ids=[first.hypothesis_id, second.hypothesis_id],
                    top_score=top_score,
                    score_margin=margin,
                    target_time=target_time,
                    scores=scores,
                    reason_codes=["disjoint_validity_intervals"],
                )

        if len(ranked) > 1 and self.config.use_decision_margin and margin < self.config.margin_threshold:
            return RevisionDecision(
                state=RevisionState.DISPUTED,
                alternative_hypothesis_ids=[ranked[0][0], ranked[1][0]],
                top_score=top_score,
                score_margin=margin,
                target_time=target_time,
                scores=scores,
                reason_codes=["insufficient_margin"],
            )

        if top_score < self.config.support_threshold:
            return RevisionDecision(
                state=RevisionState.UNKNOWN,
                alternative_hypothesis_ids=[item[0] for item in ranked],
                top_score=top_score,
                score_margin=margin,
                target_time=target_time,
                scores=scores,
                reason_codes=["below_support_threshold"],
            )

        return RevisionDecision(
            state=RevisionState.ACTIVE,
            selected_hypothesis_id=top_id,
            supporting_evidence_ids=applicable[top_id],
            top_score=top_score,
            score_margin=margin,
            target_time=target_time,
            scores=scores,
            reason_codes=["supported_with_margin"],
        )
