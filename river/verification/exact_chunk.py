"""Exact cited source/chunk verification orchestration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from river.schemas import (
    CitationVerificationResult,
    ClaimCitation,
    ClaimVerificationResult,
    ValidityInterval,
    VerificationLabel,
)

from .base import AtomicClaim, ExactChunk, VerificationBackend, VerificationInput, stable_hash


class VerificationCache:
    """Small JSON-backed cache keyed by model revision, claim hash, and chunk hash."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self._items: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            self._items = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def make_key(model_revision: str, claim_text: str, chunk_text: str) -> str:
        return stable_hash(
            {
                "model_revision": model_revision,
                "claim_hash": stable_hash(claim_text),
                "exact_chunk_hash": stable_hash(chunk_text),
            }
        )

    def get(self, key: str) -> CitationVerificationResult | None:
        value = self._items.get(key)
        if value is None:
            return None
        result = CitationVerificationResult.model_validate(value)
        result.cached = True
        return result

    def set(self, key: str, result: CitationVerificationResult) -> None:
        payload = result.model_dump(mode="json")
        payload["cached"] = False
        self._items[key] = payload
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._items, sort_keys=True, indent=2), encoding="utf-8")


def _as_text_chunk(item: Any) -> ExactChunk | None:
    if not isinstance(item, dict):
        source_id = getattr(item, "source_id", None)
        chunk_id = getattr(item, "chunk_id", None)
        text = getattr(item, "text", "")
        metadata = item.model_dump(mode="json") if hasattr(item, "model_dump") else {}
    else:
        source_id = item.get("source_id")
        chunk_id = item.get("chunk_id") or item.get("span", {}).get("chunk_id")
        text = item.get("text", "")
        metadata = dict(item)
    if not source_id or not chunk_id:
        return None
    return ExactChunk(source_id=str(source_id), chunk_id=str(chunk_id), text=str(text), metadata=metadata)


def _parse_target_time(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _temporal_mismatch(chunk: ExactChunk, target_time: datetime | None) -> bool:
    if target_time is None:
        return False
    raw = chunk.metadata.get("valid_time")
    if raw is None:
        return False
    interval = raw if isinstance(raw, ValidityInterval) else ValidityInterval.model_validate(raw)
    return interval.contains(target_time) == 0.0


class ExactChunkVerifier:
    """Verify every claim only against its cited exact source_id/chunk_id pairs."""

    def __init__(self, backend: VerificationBackend, cache: VerificationCache | None = None):
        self.backend = backend
        self.cache = cache or VerificationCache()
        # Claim/chunk pairs actually scored by the entailment model, counted
        # cumulatively so a caller can take a per-turn delta. Cache hits and
        # temporal mismatches are excluded: neither runs the model, and a cost
        # table that counted them would overstate the work done.
        self.call_count = 0

    def verify_claims(
        self,
        claims: list[AtomicClaim | dict[str, Any]],
        retrieved_chunks: list[Any],
        target_time: Any = None,
    ) -> list[ClaimVerificationResult]:
        parsed_claims = [
            claim if isinstance(claim, AtomicClaim) else AtomicClaim.from_mapping(claim)
            for claim in claims
        ]
        target = _parse_target_time(target_time)
        chunks = {
            (chunk.source_id, chunk.chunk_id): chunk
            for chunk in (_as_text_chunk(item) for item in retrieved_chunks)
            if chunk is not None
        }

        claim_results: dict[str, ClaimVerificationResult] = {}
        pending: list[tuple[str, str, VerificationInput]] = []

        for claim in parsed_claims:
            citations: list[CitationVerificationResult] = []
            if not claim.text.strip():
                claim_results[claim.claim_id] = ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    label=VerificationLabel.NEUTRAL,
                    supported_score=0.0,
                    neutral_score=1.0,
                    reason_codes=["empty_claim"],
                )
                continue
            if not claim.citations:
                claim_results[claim.claim_id] = ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    label=VerificationLabel.NEUTRAL,
                    supported_score=0.0,
                    neutral_score=1.0,
                    reason_codes=["no_citations"],
                )
                continue

            for citation in claim.citations:
                citation = citation if isinstance(citation, ClaimCitation) else ClaimCitation.model_validate(citation)
                chunk = chunks.get((citation.source_id, citation.chunk_id))
                if chunk is None:
                    citations.append(
                        CitationVerificationResult(
                            source_id=citation.source_id,
                            chunk_id=citation.chunk_id,
                            label=VerificationLabel.NEUTRAL,
                            supported_score=0.0,
                            neutral_score=1.0,
                            verifier_id=self.backend.verifier_id,
                            reason="exact_chunk_not_found",
                        )
                    )
                    continue
                pair = VerificationInput(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    source_id=citation.source_id,
                    chunk_id=citation.chunk_id,
                    chunk_text=chunk.text,
                    target_time=target,
                    metadata=chunk.metadata,
                )
                cache_key = self.cache.make_key(self.backend.model_revision, claim.text, chunk.text)
                if _temporal_mismatch(chunk, target):
                    citations.append(
                        CitationVerificationResult(
                            source_id=citation.source_id,
                            chunk_id=citation.chunk_id,
                            label=VerificationLabel.TEMPORAL_MISMATCH,
                            supported_score=0.0,
                            neutral_score=0.0,
                            temporal_mismatch_score=1.0,
                            verifier_id=self.backend.verifier_id,
                            cache_key=cache_key,
                            reason="target_time_outside_valid_interval",
                        )
                    )
                    continue
                cached = self.cache.get(cache_key)
                if cached is not None:
                    cached.source_id = citation.source_id
                    cached.chunk_id = citation.chunk_id
                    cached.cache_key = cache_key
                    citations.append(cached)
                    continue
                pending.append((claim.claim_id, cache_key, pair))

            claim_results[claim.claim_id] = ClaimVerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                label=VerificationLabel.NEUTRAL,
                citations=citations,
                reason_codes=[],
            )

        self.call_count += len(pending)
        for start in range(0, len(pending), max(self.backend.batch_size, 1)):
            batch = pending[start:start + max(self.backend.batch_size, 1)]
            scored = self.backend.score_pairs([item[2] for item in batch])
            for (claim_id, cache_key, _pair), result in zip(batch, scored, strict=True):
                result.cache_key = cache_key
                self.cache.set(cache_key, result)
                claim_results[claim_id].citations.append(result)

        return [_aggregate_claim_result(claim_results[claim.claim_id]) for claim in parsed_claims]


def _aggregate_claim_result(result: ClaimVerificationResult) -> ClaimVerificationResult:
    if not result.citations:
        return result

    supported = [item for item in result.citations if item.label == VerificationLabel.SUPPORTED]
    contradicted = [item for item in result.citations if item.label == VerificationLabel.CONTRADICTED]
    temporal = [item for item in result.citations if item.label == VerificationLabel.TEMPORAL_MISMATCH]
    neutral = [item for item in result.citations if item.label == VerificationLabel.NEUTRAL]

    if supported:
        best = max(supported, key=lambda item: item.supported_score or 0.0)
        result.label = VerificationLabel.SUPPORTED
        result.supported_score = best.supported_score
        result.neutral_score = best.neutral_score
        result.contradicted_score = best.contradicted_score
        result.temporal_mismatch_score = best.temporal_mismatch_score
        result.selected_source_id = best.source_id
        result.selected_chunk_id = best.chunk_id
        result.reason_codes = ["at_least_one_exact_citation_supported"]
        return result
    if contradicted:
        best = max(contradicted, key=lambda item: item.contradicted_score or 0.0)
        result.label = VerificationLabel.CONTRADICTED
        result.contradicted_score = best.contradicted_score
        result.supported_score = best.supported_score
        result.neutral_score = best.neutral_score
        result.reason_codes = ["exact_citations_contradict"]
        return result
    if temporal and not neutral:
        result.label = VerificationLabel.TEMPORAL_MISMATCH
        result.temporal_mismatch_score = 1.0
        result.supported_score = 0.0
        result.reason_codes = ["all_exact_citations_temporally_mismatched"]
        return result

    best_neutral = max(result.citations, key=lambda item: item.neutral_score or 0.0)
    result.label = VerificationLabel.NEUTRAL
    result.supported_score = best_neutral.supported_score
    result.neutral_score = best_neutral.neutral_score
    result.contradicted_score = best_neutral.contradicted_score
    result.temporal_mismatch_score = best_neutral.temporal_mismatch_score
    result.reason_codes = ["no_exact_citation_supported"]
    return result
