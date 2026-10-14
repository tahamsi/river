"""MiniCheck exact-chunk verifier baseline without RIVER revision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from river.schemas import ClaimCitation, VerificationLabel
from river.verification import AtomicClaim, ExactChunkVerifier, MiniCheckConfig, MiniCheckVerifier, VerificationCache
from river.verification.base import VerifierUnavailable
from river.verification.calibration import calibrate_support_threshold

from .base import Baseline, BaselineCapabilities, BaselineUnavailable
from .native import NativeBaselineManifest


class MiniCheckNoRevisionBaseline(Baseline):
    name = "minicheck_no_revision"
    version = "official-minicheck-b58b9fa-flan-t5-large-96eafd0"
    capabilities = BaselineCapabilities(
        persistent_memory=False,
        temporal_validity=False,
        citation_verification=True,
        conflict_resolution=False,
        abstention=True,
        native_external_system=True,
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        verifier_config = dict(self.config.get("verifier", {}))
        scorer = verifier_config.pop("scorer", None)
        cache_path = verifier_config.pop("verification_cache_path", None)
        threshold = self._threshold_from_calibration(verifier_config)
        if threshold is not None:
            verifier_config["support_threshold"] = threshold
        try:
            minicheck_config = MiniCheckConfig(**{
                key: value
                for key, value in verifier_config.items()
                if key in MiniCheckConfig.__dataclass_fields__
            })
            backend = MiniCheckVerifier(config=minicheck_config, scorer=scorer)
        except VerifierUnavailable as exc:
            raise BaselineUnavailable(str(exc)) from exc
        cache = VerificationCache(Path(cache_path)) if cache_path else VerificationCache()
        self.verifier = ExactChunkVerifier(backend, cache=cache)
        self._verifier_backend = backend

    def _threshold_from_calibration(self, verifier_config: dict[str, Any]) -> float | None:
        calibration_rows = verifier_config.get("calibration_rows")
        if calibration_rows:
            return calibrate_support_threshold(calibration_rows)
        calibration_file = verifier_config.get("calibration_file")
        if calibration_file:
            import json

            rows = [
                json.loads(line)
                for line in Path(calibration_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return calibrate_support_threshold(rows)
        return None

    def manifest(self) -> dict[str, Any]:
        data = super().manifest()
        data["native_manifest"] = NativeBaselineManifest.load(self.name).as_dict()
        return data

    def process_turn(self, session_id, query, retrieved_chunks, target_time=None):
        del session_id
        calls_before = self.verifier.call_count
        claims = self._generate_atomic_claims(query, retrieved_chunks)
        verification = self.verifier.verify_claims(claims, retrieved_chunks, target_time)
        accepted = all(item.label == VerificationLabel.SUPPORTED for item in verification) and bool(verification)
        if accepted:
            answer = " ".join(item.claim_text for item in verification)
            return self._result("active", answer, verification, retried=False, calls_before=calls_before)

        supported_claims = [
            claim
            for claim, result in zip(claims, verification, strict=True)
            if result.label == VerificationLabel.SUPPORTED
        ]
        if supported_claims:
            retry_verification = self.verifier.verify_claims(supported_claims, retrieved_chunks, target_time)
            answer = " ".join(item.text for item in supported_claims)
            return self._result("active", answer, retry_verification, retried=True, calls_before=calls_before)

        abstain = "I do not have sufficient support from the cited exact chunks."
        return self._result("abstain", abstain, verification, retried=True, calls_before=calls_before)

    def _result(self, state: str, answer: str, verification, *, retried: bool, calls_before: int) -> dict[str, Any]:
        dumped = [item.model_dump(mode="json") for item in verification]
        return {
            "answer": answer,
            "predicted_state": state,
            "predicted_hypothesis_id": None,
            "answer_correct": None,
            "citation_supported": state == "active",
            "claim_verification_results": dumped,
            # This baseline answers by joining its own cited chunks, so it calls
            # no generator at all. Zero calls and zero generated tokens are
            # measurements here, not absent telemetry.
            "input_tokens": 0,
            "output_tokens": 0,
            "verification": {
                "pipeline": "retrieve_generate_atomic_claims_exact_chunk_minicheck_accept_retry_abstain",
                "retried": retried,
                "verifier": self._verifier_backend.manifest(),
                "nli_calls": self.verifier.call_count - calls_before,
                "model_calls": 0,
            },
        }

    def _generate_atomic_claims(self, query: str, retrieved_chunks: list[Any]) -> list[AtomicClaim]:
        configured = self.config.get("claims")
        if configured is not None:
            return [claim if isinstance(claim, AtomicClaim) else AtomicClaim.from_mapping(claim) for claim in configured]

        claims: list[AtomicClaim] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump(mode="json") if hasattr(chunk, "model_dump") else vars(chunk)
            source_id = chunk.get("source_id")
            chunk_id = chunk.get("chunk_id") or chunk.get("span", {}).get("chunk_id")
            if not source_id or not chunk_id:
                continue
            claim_text = chunk.get("claim") or self._claim_from_structured_chunk(chunk) or self._claim_from_text(chunk)
            if not claim_text:
                continue
            claims.append(
                AtomicClaim(
                    claim_id=str(chunk.get("claim_id", f"claim-{index}")),
                    text=claim_text,
                    citations=[ClaimCitation(source_id=str(source_id), chunk_id=str(chunk_id))],
                )
            )
        if not claims and query.strip():
            claims.append(AtomicClaim(claim_id="claim-1", text="", citations=[]))
        return claims

    @staticmethod
    def _claim_from_structured_chunk(chunk: dict[str, Any]) -> str | None:
        if {"subject", "predicate", "object"} <= chunk.keys():
            predicate = str(chunk["predicate"]).replace("_", " ")
            return f"{chunk['subject']} {predicate} {chunk['object']}."
        return None

    @staticmethod
    def _claim_from_text(chunk: dict[str, Any]) -> str:
        text = str(chunk.get("text", "")).strip()
        if not text:
            return ""
        first_sentence = text.split(".")[0].strip()
        return first_sentence + "." if first_sentence else text
