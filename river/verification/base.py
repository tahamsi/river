"""Shared primitives for claim verification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from river.schemas import ClaimCitation, CitationVerificationResult, VerificationLabel


class VerifierUnavailable(RuntimeError):
    """Raised when an optional verifier dependency/model is unavailable."""


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def stable_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    text: str
    citations: list[ClaimCitation]

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "AtomicClaim":
        supports = item.get("supports", item.get("citations", []))
        citations = [
            support if isinstance(support, ClaimCitation) else ClaimCitation.model_validate(support)
            for support in supports
        ]
        return cls(
            claim_id=str(item.get("id", item.get("claim_id", "claim"))),
            text=str(item.get("text", item.get("claim", ""))),
            citations=citations,
        )


@dataclass(frozen=True)
class ExactChunk:
    source_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VerificationInput:
    claim_id: str
    claim_text: str
    source_id: str
    chunk_id: str
    chunk_text: str
    target_time: datetime | None = None
    metadata: dict[str, Any] | None = None


class VerificationBackend(Protocol):
    verifier_id: str
    model_revision: str
    batch_size: int
    max_input_length: int | None

    def score_pairs(self, pairs: list[VerificationInput]) -> list[CitationVerificationResult]:
        ...


def empty_result(pair: VerificationInput, verifier_id: str, reason: str) -> CitationVerificationResult:
    return CitationVerificationResult(
        source_id=pair.source_id,
        chunk_id=pair.chunk_id,
        label=VerificationLabel.NEUTRAL,
        supported_score=0.0,
        neutral_score=1.0,
        verifier_id=verifier_id,
        reason=reason,
    )
