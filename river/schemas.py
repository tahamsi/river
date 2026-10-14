"""Stable schemas shared by RIVER, baselines, evaluation, and table generation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RevisionState(StrEnum):
    ACTIVE = "active"
    DISPUTED = "disputed"
    EVOLUTION = "evolution"
    UNKNOWN = "unknown"
    ABSTAIN = "abstain"


class VerificationLabel(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    ERROR = "error"


class ComparisonMode(StrEnum):
    CONTROLLED = "controlled"
    NATIVE = "native"


class ValidityInterval(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def contains(self, target: datetime | None) -> float:
        if target is None:
            return self.confidence
        if self.start is not None and target < self.start:
            return 0.0
        if self.end is not None and target > self.end:
            return 0.0
        return self.confidence

    def overlaps(self, other: "ValidityInterval") -> bool:
        if self.end is not None and other.start is not None and self.end < other.start:
            return False
        if other.end is not None and self.start is not None and other.end < self.start:
            return False
        return True


class EvidenceRecord(BaseModel):
    evidence_id: str
    source_id: str
    chunk_id: str
    text: str = ""
    observed_at: datetime
    valid_time: ValidityInterval = Field(default_factory=ValidityInterval)
    entailment: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance_reliability: float = Field(default=1.0, ge=0.0, le=1.0)
    independence_group: str


class BeliefHypothesis(BaseModel):
    hypothesis_id: str
    subject: str
    predicate: str
    object: str
    evidence_ids: list[str] = Field(default_factory=list)


class RevisionDecision(BaseModel):
    state: RevisionState
    selected_hypothesis_id: str | None = None
    alternative_hypothesis_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    top_score: float | None = None
    score_margin: float | None = None
    target_time: datetime | None = None
    reason_codes: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class ClaimCitation(BaseModel):
    source_id: str
    chunk_id: str


class CitationVerificationResult(BaseModel):
    source_id: str
    chunk_id: str
    label: VerificationLabel
    supported_score: float | None = Field(default=None, ge=0.0, le=1.0)
    contradicted_score: float | None = Field(default=None, ge=0.0, le=1.0)
    neutral_score: float | None = Field(default=None, ge=0.0, le=1.0)
    temporal_mismatch_score: float | None = Field(default=None, ge=0.0, le=1.0)
    verifier_id: str
    cache_key: str | None = None
    cached: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimVerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    label: VerificationLabel
    supported_score: float | None = Field(default=None, ge=0.0, le=1.0)
    contradicted_score: float | None = Field(default=None, ge=0.0, le=1.0)
    neutral_score: float | None = Field(default=None, ge=0.0, le=1.0)
    temporal_mismatch_score: float | None = Field(default=None, ge=0.0, le=1.0)
    citations: list[CitationVerificationResult] = Field(default_factory=list)
    selected_source_id: str | None = None
    selected_chunk_id: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class PredictionRecord(BaseModel):
    run_id: str
    example_id: str
    base_example_id: str
    pair_id: str | None = None
    dataset: str
    source_dataset: str
    split: str
    intervention: str
    comparison_mode: ComparisonMode = ComparisonMode.CONTROLLED
    baseline: str
    baseline_version: str
    generator: str
    seed: int
    gold_state: str | None = None
    gold_hypothesis_id: str | None = None
    predicted_state: str | None = None
    predicted_hypothesis_id: str | None = None
    answer: str | None = None
    intervention_hypothesis_id: str | None = None
    answer_correct: bool | None = None
    # Whether the explicit FINAL_ANSWER field exactly matches the gold answer
    # or a recorded alias. None where no single answer is correct.
    answer_text_correct: bool | None = None
    # Parsed value from the first-line FINAL_ANSWER field.
    direct_answer: str | None = None
    # Which hypothesis the explicit direct answer names.
    answer_named_hypothesis_id: str | None = None
    # Whether the output follows the active/non-active rendering contract.
    answer_contract_valid: bool | None = None
    # Whether the rendered direct answer agrees with the operator selection.
    answer_decision_faithful: bool | None = None
    # Requires correct active state, hypothesis, direct answer and fidelity.
    end_to_end_answer_correct: bool | None = None
    citation_supported: bool | None = None
    confidence: float | None = None
    state_probabilities: dict[str, float] | None = None
    claim_verification_results: list[ClaimVerificationResult] | None = None
    claim_verification_result_refs: list[str] | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    configuration_hash: str
    code_commit: str
    dataset_hash: str
    prompt_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
