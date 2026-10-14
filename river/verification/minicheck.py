"""Revision-pinned MiniCheck Flan-T5 verifier backend.

The upstream MiniCheck class at the pinned package commit hard-codes the model
repository and calls ``from_pretrained`` without a model revision. Recording a
revision in RIVER metadata therefore did not prove which weights were loaded.
This adapter reproduces the official Flan-T5 exact-chunk scoring path while
loading an explicit Hugging Face snapshot directory and failing closed if the
resolved commit differs from the configured revision.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from river.schemas import CitationVerificationResult, VerificationLabel

from .base import VerificationInput, VerifierUnavailable


OFFICIAL_MINICHECK_PACKAGE_VERSION = "0.1.0"
OFFICIAL_MINICHECK_PACKAGE_COMMIT = "b58b9fa69acbd1015ec970fa65dd752413a053d2"
OFFICIAL_MINICHECK_PACKAGE_SPEC = (
    "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@"
    f"{OFFICIAL_MINICHECK_PACKAGE_COMMIT}"
)
OFFICIAL_MINICHECK_REPOSITORY_LICENSE = "Apache-2.0"
OFFICIAL_MINICHECK_MODEL_NAME = "flan-t5-large"
OFFICIAL_MINICHECK_MODEL_IDENTIFIER = "lytang/MiniCheck-Flan-T5-Large"
OFFICIAL_MINICHECK_MODEL_REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
OFFICIAL_MINICHECK_MODEL_LICENSE = "MIT"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class MiniCheckConfig:
    package_spec: str = OFFICIAL_MINICHECK_PACKAGE_SPEC
    package_version: str = OFFICIAL_MINICHECK_PACKAGE_VERSION
    package_commit: str = OFFICIAL_MINICHECK_PACKAGE_COMMIT
    repository_license: str = OFFICIAL_MINICHECK_REPOSITORY_LICENSE
    model_name: str = OFFICIAL_MINICHECK_MODEL_NAME
    model_identifier: str = OFFICIAL_MINICHECK_MODEL_IDENTIFIER
    model_revision: str = OFFICIAL_MINICHECK_MODEL_REVISION
    model_license: str = OFFICIAL_MINICHECK_MODEL_LICENSE
    inference_device: str = "auto"
    batch_size: int = 16
    max_input_length: int = 2048
    support_threshold: float = 0.5
    cache_dir: str | None = ".cache/minicheck-pinned"
    local_files_only: bool = False

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


class PinnedMiniCheckFlanT5Scorer:
    """Exact-chunk scorer matching MiniCheck's official Flan-T5 label rule."""

    unsupported_token_id = 3
    supported_token_id = 209

    def __init__(self, config: MiniCheckConfig):
        try:
            import torch
            import transformers
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional environment
            raise VerifierUnavailable(
                "Pinned MiniCheck requires the minicheck optional environment "
                "(transformers, torch, accelerate and huggingface-hub)."
            ) from exc

        cache_dir = str(Path(config.cache_dir)) if config.cache_dir else None
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=config.model_identifier,
                    revision=config.model_revision,
                    cache_dir=cache_dir,
                    local_files_only=config.local_files_only,
                )
            ).resolve()
        except Exception as exc:  # pragma: no cover - network/cache dependent
            raise VerifierUnavailable(
                f"Could not resolve {config.model_identifier}@{config.model_revision}"
            ) from exc

        resolved_revision = snapshot.name
        if resolved_revision != config.model_revision:
            raise VerifierUnavailable(
                "MiniCheck snapshot mismatch: requested "
                f"{config.model_revision}, resolved {resolved_revision}"
            )

        weight_files = sorted(
            path for pattern in ("*.safetensors", "*.bin") for path in snapshot.glob(pattern)
        )
        if not weight_files:
            raise VerifierUnavailable(f"No model weight file found in pinned snapshot {snapshot}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                snapshot,
                local_files_only=True,
                device_map=config.inference_device,
            )
        except Exception as exc:  # pragma: no cover - model environment dependent
            raise VerifierUnavailable(f"Could not load pinned MiniCheck snapshot {snapshot}") from exc
        self.model.eval()
        self._torch = torch
        self.batch_size = config.batch_size
        self.max_input_length = config.max_input_length
        self.provenance = {
            "resolved_model_revision": resolved_revision,
            "weight_files": {path.name: _sha256(path) for path in weight_files},
            "transformers_version": transformers.__version__,
            "load_source": "explicit_local_snapshot",
        }

    def score(self, docs, claims, chunk_size=None):
        if chunk_size is not None:
            raise ValueError("RIVER MiniCheck uses exact chunks; chunk_size must be None")
        if len(docs) != len(claims):
            raise ValueError("docs and claims must have equal length")

        probabilities: list[float] = []
        eos = self.tokenizer.eos_token or "</s>"
        for start in range(0, len(docs), self.batch_size):
            doc_batch = docs[start : start + self.batch_size]
            claim_batch = claims[start : start + self.batch_size]
            texts = [f"predict: {doc}{eos}{claim}" for doc, claim in zip(doc_batch, claim_batch, strict=True)]
            encoded = self.tokenizer(
                texts,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            device = self.model.device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            decoder_input_ids = self._torch.zeros(
                (len(texts), 1), dtype=self._torch.long, device=device
            )
            with self._torch.inference_mode():
                output = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                )
            label_ids = self._torch.tensor(
                [self.unsupported_token_id, self.supported_token_id],
                device=output.logits.device,
            )
            label_logits = output.logits[:, 0, :].index_select(1, label_ids)
            batch_probs = self._torch.softmax(label_logits, dim=-1)[:, 1].detach().cpu().tolist()
            probabilities.extend(float(value) for value in batch_probs)

        labels = [1 if value > 0.5 else 0 for value in probabilities]
        return labels, probabilities, list(docs), [[value] for value in probabilities]


class MiniCheckVerifier:
    """MiniCheck exact-chunk verifier with enforced model provenance."""

    verifier_id = "official_minicheck_flan_t5_large_exact_chunk_pinned"

    def __init__(self, config: MiniCheckConfig | None = None, scorer: Any | None = None):
        self.config = config or MiniCheckConfig()
        self.batch_size = self.config.batch_size
        self.max_input_length = self.config.max_input_length
        self._scorer = scorer or PinnedMiniCheckFlanT5Scorer(self.config)
        provenance = getattr(self._scorer, "provenance", {})
        self.model_revision = provenance.get("resolved_model_revision", self.config.model_revision)
        if scorer is None and self.model_revision != self.config.model_revision:
            raise VerifierUnavailable(
                f"MiniCheck loaded {self.model_revision}, expected {self.config.model_revision}"
            )

    def manifest(self) -> dict[str, Any]:
        manifest = self.config.manifest()
        manifest["verifier_id"] = self.verifier_id
        manifest["resolved_provenance"] = getattr(self._scorer, "provenance", None)
        return manifest

    def score_pairs(self, pairs: list[VerificationInput]) -> list[CitationVerificationResult]:
        if not pairs:
            return []
        docs = [pair.chunk_text for pair in pairs]
        claims = [pair.claim_text for pair in pairs]
        pred_labels, raw_probs, _used_chunks, _support_prob_per_chunk = self._scorer.score(
            docs=docs,
            claims=claims,
            chunk_size=None,
        )
        results: list[CitationVerificationResult] = []
        resolved = getattr(self._scorer, "provenance", None)
        for pair, pred_label, raw_prob in zip(pairs, pred_labels, raw_probs, strict=True):
            score = float(raw_prob)
            label = (
                VerificationLabel.SUPPORTED
                if int(pred_label) == 1 and score >= self.config.support_threshold
                else VerificationLabel.NEUTRAL
            )
            results.append(
                CitationVerificationResult(
                    source_id=pair.source_id,
                    chunk_id=pair.chunk_id,
                    label=label,
                    supported_score=score,
                    neutral_score=max(0.0, 1.0 - score),
                    verifier_id=self.verifier_id,
                    reason="minicheck_supported_probability",
                    metadata={
                        "package_commit": self.config.package_commit,
                        "model_identifier": self.config.model_identifier,
                        "requested_model_revision": self.config.model_revision,
                        "resolved_provenance": resolved,
                        "model_license": self.config.model_license,
                        "repository_license": self.config.repository_license,
                        "inference_device": self.config.inference_device,
                        "max_input_length": self.config.max_input_length,
                    },
                )
            )
        return results
