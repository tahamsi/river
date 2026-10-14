"""Lightweight lexical and embedding-similarity verifier backends."""

from __future__ import annotations

import math
from collections import Counter

from river.schemas import CitationVerificationResult, VerificationLabel

from .base import VerificationInput, tokenize


class LexicalCoverageVerifier:
    verifier_id = "lexical_coverage_v1"
    model_revision = "lexical_coverage_v1"
    max_input_length = None

    def __init__(self, coverage_threshold: float = 0.7, batch_size: int = 64):
        self.coverage_threshold = coverage_threshold
        self.batch_size = batch_size

    def score_pairs(self, pairs: list[VerificationInput]) -> list[CitationVerificationResult]:
        results: list[CitationVerificationResult] = []
        for pair in pairs:
            claim_tokens = set(tokenize(pair.claim_text))
            chunk_tokens = set(tokenize(pair.chunk_text))
            if not claim_tokens:
                score = 0.0
            else:
                score = len(claim_tokens & chunk_tokens) / len(claim_tokens)
            label = VerificationLabel.SUPPORTED if score >= self.coverage_threshold else VerificationLabel.NEUTRAL
            results.append(
                CitationVerificationResult(
                    source_id=pair.source_id,
                    chunk_id=pair.chunk_id,
                    label=label,
                    supported_score=score,
                    neutral_score=1.0 - score,
                    verifier_id=self.verifier_id,
                    reason="lexical_coverage",
                )
            )
        return results


class EmbeddingSimilarityVerifier:
    verifier_id = "hash_embedding_similarity_v1"
    model_revision = "hash_embedding_similarity_v1"
    max_input_length = None

    def __init__(self, similarity_threshold: float = 0.75, batch_size: int = 64, dimensions: int = 128):
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token, count in Counter(tokenize(text)).items():
            values[hash(token) % self.dimensions] += float(count)
        return values

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def score_pairs(self, pairs: list[VerificationInput]) -> list[CitationVerificationResult]:
        results: list[CitationVerificationResult] = []
        for pair in pairs:
            score = self._cosine(self._embed(pair.claim_text), self._embed(pair.chunk_text))
            label = VerificationLabel.SUPPORTED if score >= self.similarity_threshold else VerificationLabel.NEUTRAL
            results.append(
                CitationVerificationResult(
                    source_id=pair.source_id,
                    chunk_id=pair.chunk_id,
                    label=label,
                    supported_score=score,
                    neutral_score=1.0 - score,
                    verifier_id=self.verifier_id,
                    reason="embedding_similarity",
                )
            )
        return results
