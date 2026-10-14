"""Optional generic NLI verifier backend."""

from __future__ import annotations

from typing import Any

from river.schemas import CitationVerificationResult, VerificationLabel

from .base import VerificationInput, VerifierUnavailable


class GenericNLIVerifier:
    """Generic HuggingFace NLI interface, loaded only when configured."""

    verifier_id = "generic_nli_exact_chunk"

    def __init__(
        self,
        model_identifier: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        model_revision: str = "main",
        batch_size: int = 16,
        max_input_length: int = 512,
        pipeline: Any | None = None,
    ):
        self.model_identifier = model_identifier
        self.model_revision = model_revision
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self._pipeline = pipeline
        if self._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
            except Exception as exc:
                raise VerifierUnavailable("Install transformers/torch to use the generic NLI verifier") from exc
            self._pipeline = hf_pipeline(
                "text-classification",
                model=model_identifier,
                revision=model_revision,
                truncation=True,
                max_length=max_input_length,
            )

    def score_pairs(self, pairs: list[VerificationInput]) -> list[CitationVerificationResult]:
        inputs = [
            {"text": pair.chunk_text, "text_pair": pair.claim_text}
            for pair in pairs
        ]
        raw = self._pipeline(inputs)
        results: list[CitationVerificationResult] = []
        for pair, output in zip(pairs, raw, strict=True):
            label_text = str(output.get("label", "")).lower()
            score = float(output.get("score", 0.0))
            if "entail" in label_text:
                label = VerificationLabel.SUPPORTED
                supported, contradicted, neutral = score, 0.0, 1.0 - score
            elif "contrad" in label_text:
                label = VerificationLabel.CONTRADICTED
                supported, contradicted, neutral = 0.0, score, 1.0 - score
            else:
                label = VerificationLabel.NEUTRAL
                supported, contradicted, neutral = 0.0, 0.0, score
            results.append(
                CitationVerificationResult(
                    source_id=pair.source_id,
                    chunk_id=pair.chunk_id,
                    label=label,
                    supported_score=supported,
                    contradicted_score=contradicted,
                    neutral_score=neutral,
                    verifier_id=self.verifier_id,
                    reason="generic_nli",
                    metadata={
                        "model_identifier": self.model_identifier,
                        "model_revision": self.model_revision,
                    },
                )
            )
        return results
