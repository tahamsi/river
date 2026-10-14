"""Validated JSONL schema for RIVER-Bench examples."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from river.schemas import BeliefHypothesis, RevisionState, ValidityInterval


class BenchmarkEvidence(BaseModel):
    """Evidence with transaction time separated from world-valid time."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_id: str
    chunk_id: str
    text: str
    subject: str
    predicate: str
    object: str
    hypothesis_id: str
    transaction_time: datetime
    valid_time: ValidityInterval
    entailment: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance_reliability: float = Field(default=1.0, ge=0.0, le=1.0)
    independence_group: str
    verifier_scores_measured_by: str | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("evidence_id", "source_id", "chunk_id", "hypothesis_id", "independence_group")
    @classmethod
    def non_empty_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("identifier fields must be non-empty")
        return value

    @model_validator(mode="after")
    def valid_interval_is_ordered(self) -> "BenchmarkEvidence":
        start = self.valid_time.start
        end = self.valid_time.end
        if start is not None and end is not None and start > end:
            raise ValueError(f"{self.evidence_id} has valid_time.start after valid_time.end")
        return self

    def as_river_record(self) -> dict[str, Any]:
        """Return the existing ``river.schemas.EvidenceRecord`` shape."""

        return {
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "observed_at": self.transaction_time,
            "valid_time": self.valid_time.model_dump(mode="json"),
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral,
            "extraction_confidence": self.extraction_confidence,
            "provenance_reliability": self.provenance_reliability,
            "independence_group": self.independence_group,
        }


class BenchmarkExample(BaseModel):
    """One normalized RIVER-Bench JSONL example."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    base_example_id: str
    pair_id: str
    dataset: str
    source_dataset: str
    split: str
    intervention: str
    session_id: str
    query: str
    target_time: datetime | None = None
    evidence: list[BenchmarkEvidence]
    hypotheses: list[BeliefHypothesis]
    gold_state: RevisionState
    gold_hypothesis_id: str | None = None
    intervention_hypothesis_id: str | None = None
    transformation_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "example_id",
        "base_example_id",
        "pair_id",
        "dataset",
        "source_dataset",
        "split",
        "intervention",
        "session_id",
        "query",
    )
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required string fields must be non-empty")
        return value

    @model_validator(mode="after")
    def references_are_consistent(self) -> "BenchmarkExample":
        evidence_ids = [item.evidence_id for item in self.evidence]
        duplicate_evidence = [value for value, count in Counter(evidence_ids).items() if count > 1]
        if duplicate_evidence:
            raise ValueError(f"duplicate evidence IDs: {duplicate_evidence}")

        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        duplicate_hypotheses = [value for value, count in Counter(hypothesis_ids).items() if count > 1]
        if duplicate_hypotheses:
            raise ValueError(f"duplicate hypothesis IDs: {duplicate_hypotheses}")
        hypothesis_id_set = set(hypothesis_ids)

        for item in self.evidence:
            if item.hypothesis_id not in hypothesis_id_set:
                raise ValueError(f"{item.evidence_id} references unknown hypothesis {item.hypothesis_id}")

        evidence_id_set = set(evidence_ids)
        for hypothesis in self.hypotheses:
            missing = sorted(set(hypothesis.evidence_ids) - evidence_id_set)
            if missing:
                raise ValueError(f"{hypothesis.hypothesis_id} references missing evidence IDs {missing}")

        if self.gold_hypothesis_id is not None and self.gold_hypothesis_id not in hypothesis_id_set:
            raise ValueError(f"gold_hypothesis_id {self.gold_hypothesis_id} is not in hypotheses")
        if (
            self.intervention_hypothesis_id is not None
            and self.intervention_hypothesis_id not in hypothesis_id_set
        ):
            raise ValueError(
                f"intervention_hypothesis_id {self.intervention_hypothesis_id} is not in hypotheses"
            )
        return self

    def evidence_for_runner(self) -> list[dict[str, Any]]:
        return [item.as_river_record() for item in self.evidence]

    def hypotheses_for_runner(self) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.hypotheses]
