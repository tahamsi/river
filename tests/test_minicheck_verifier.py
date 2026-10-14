import os

import pytest

from river import build_baseline
from river.baselines import BaselineUnavailable
from river.schemas import ClaimCitation, VerificationLabel
from river.verification import AtomicClaim, ExactChunkVerifier, MiniCheckConfig, MiniCheckVerifier, VerificationCache
from river.verification.base import VerifierUnavailable
from river.verification.calibration import calibrate_support_threshold
import river.verification.minicheck as minicheck_module


class MockMiniCheckScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0
        self.seen = []

    def score(self, docs, claims, chunk_size=None):
        del chunk_size
        self.calls += 1
        self.seen.extend(zip(docs, claims, strict=True))
        raw = [float(self.scores.get((doc, claim), 0.05)) for doc, claim in zip(docs, claims, strict=True)]
        labels = [1 if score >= 0.5 else 0 for score in raw]
        return labels, raw, docs, [[score] for score in raw]


def chunk(source_id, chunk_id, text, **extra):
    payload = {"source_id": source_id, "chunk_id": chunk_id, "text": text}
    payload.update(extra)
    return payload


def claim(claim_id, text, *citations):
    return AtomicClaim(
        claim_id=claim_id,
        text=text,
        citations=[ClaimCitation(source_id=source_id, chunk_id=chunk_id) for source_id, chunk_id in citations],
    )


def verifier(scores, *, revision="mock-model-revision"):
    config = MiniCheckConfig(model_revision=revision, batch_size=2, max_input_length=128)
    return ExactChunkVerifier(MiniCheckVerifier(config=config, scorer=MockMiniCheckScorer(scores)))


def test_correct_chunk_supports_claim():
    text = "The reactor is online."
    target = claim("c1", "The reactor is online.", ("doc", "chunk-a"))
    results = verifier({(text, target.text): 0.92}).verify_claims([target], [chunk("doc", "chunk-a", text)])

    assert results[0].label == VerificationLabel.SUPPORTED
    assert results[0].selected_source_id == "doc"
    assert results[0].selected_chunk_id == "chunk-a"


def test_wrong_chunk_from_correct_document_fails():
    good = "The reactor is online."
    wrong = "The reactor maintenance window is tomorrow."
    target = claim("c1", "The reactor is online.", ("doc", "chunk-b"))
    results = verifier({(good, target.text): 0.95}).verify_claims(
        [target],
        [chunk("doc", "chunk-a", good), chunk("doc", "chunk-b", wrong)],
    )

    assert results[0].label == VerificationLabel.NEUTRAL
    assert results[0].selected_chunk_id is None


def test_negation_flip_fails():
    text = "The safety valve is open."
    target = claim("c1", "The safety valve is not open.", ("doc", "chunk-a"))
    results = verifier({(text, target.text): 0.04}).verify_claims([target], [chunk("doc", "chunk-a", text)])

    assert results[0].label != VerificationLabel.SUPPORTED


def test_modality_flip_fails():
    text = "The lab may use Protocol Beta after approval."
    target = claim("c1", "The lab must use Protocol Beta.", ("doc", "chunk-a"))
    results = verifier({(text, target.text): 0.18}).verify_claims([target], [chunk("doc", "chunk-a", text)])

    assert results[0].label != VerificationLabel.SUPPORTED


def test_numerical_mismatch_fails():
    text = "The invoice total is 300 credits."
    target = claim("c1", "The invoice total is 30 credits.", ("doc", "chunk-a"))
    results = verifier({(text, target.text): 0.09}).verify_claims([target], [chunk("doc", "chunk-a", text)])

    assert results[0].label == VerificationLabel.NEUTRAL


def test_missing_chunk_fails_without_document_fallback():
    target = claim("c1", "The archive opened in 1986.", ("doc", "missing"))
    backend = verifier({("The archive opened in 1986.", target.text): 0.99})
    results = backend.verify_claims([target], [chunk("doc", "other", "The archive opened in 1986.")])

    assert results[0].label == VerificationLabel.NEUTRAL
    assert results[0].citations[0].reason == "exact_chunk_not_found"


def test_empty_claim_fails():
    target = claim("c1", "", ("doc", "chunk-a"))
    results = verifier({}).verify_claims([target], [chunk("doc", "chunk-a", "Anything.")])

    assert results[0].label == VerificationLabel.NEUTRAL
    assert results[0].reason_codes == ["empty_claim"]


def test_multiple_citations_where_one_supports():
    bad = "The source says only that a meeting happened."
    good = "The committee approved the budget."
    target = claim("c1", "The committee approved the budget.", ("doc", "bad"), ("doc", "good"))
    results = verifier({(bad, target.text): 0.03, (good, target.text): 0.97}).verify_claims(
        [target],
        [chunk("doc", "bad", bad), chunk("doc", "good", good)],
    )

    assert results[0].label == VerificationLabel.SUPPORTED
    assert results[0].selected_chunk_id == "good"
    assert len(results[0].citations) == 2


def test_model_unavailable_error(monkeypatch):
    def unavailable(_config):
        raise VerifierUnavailable("weights unavailable")

    monkeypatch.setattr(minicheck_module, "PinnedMiniCheckFlanT5Scorer", unavailable)
    with pytest.raises(VerifierUnavailable):
        MiniCheckVerifier(scorer=None)
    with pytest.raises(BaselineUnavailable):
        build_baseline("minicheck_no_revision", {"verifier": {}})


def test_cache_identity():
    text = "The route is N4."
    target = claim("c1", "The route is N4.", ("doc", "chunk-a"))
    scorer = MockMiniCheckScorer({(text, target.text): 0.88})
    cache = VerificationCache()
    backend = MiniCheckVerifier(
        config=MiniCheckConfig(model_revision="revision-a", batch_size=4),
        scorer=scorer,
    )
    exact = ExactChunkVerifier(backend, cache=cache)

    first = exact.verify_claims([target], [chunk("doc", "chunk-a", text)])[0]
    second = exact.verify_claims([target], [chunk("doc", "chunk-a", text)])[0]
    same_pair_key = VerificationCache.make_key("revision-a", target.text, text)
    changed_model_key = VerificationCache.make_key("revision-b", target.text, text)

    assert scorer.calls == 1
    assert first.citations[0].cache_key == same_pair_key
    assert second.citations[0].cached is True
    assert same_pair_key != changed_model_key


def test_manifest_distinguishes_requested_and_resolved_provenance():
    scorer = MockMiniCheckScorer({})
    scorer.provenance = {
        "resolved_model_revision": "revision-a",
        "weight_files": {"model.safetensors": "sha256:abc"},
        "load_source": "explicit_local_snapshot",
    }
    backend = MiniCheckVerifier(
        config=MiniCheckConfig(model_revision="revision-a"),
        scorer=scorer,
    )

    manifest = backend.manifest()
    assert manifest["model_revision"] == "revision-a"
    assert manifest["resolved_provenance"]["resolved_model_revision"] == "revision-a"
    assert manifest["resolved_provenance"]["weight_files"]


def test_minicheck_no_revision_baseline_uses_exact_chunk_verification():
    text = "The sensor mode is manual."
    target = "The sensor mode is manual."
    scorer = MockMiniCheckScorer({(text, target): 0.91})
    baseline = build_baseline(
        "minicheck_no_revision",
        {
            "verifier": {"scorer": scorer, "model_revision": "mock"},
            "claims": [
                {
                    "id": "c1",
                    "text": target,
                    "supports": [{"source_id": "doc", "chunk_id": "chunk-a"}],
                }
            ],
        },
    )

    result = baseline.process_turn("session", "query", [chunk("doc", "chunk-a", text)])

    assert result["predicted_state"] == "active"
    assert result["citation_supported"] is True
    assert result["verification"]["pipeline"] == "retrieve_generate_atomic_claims_exact_chunk_minicheck_accept_retry_abstain"


def test_calibration_rejects_test_rows():
    rows = [
        {"split": "validation", "supported_score": 0.8, "label": "supported"},
        {"split": "test", "supported_score": 0.2, "label": "neutral"},
    ]

    with pytest.raises(ValueError, match="train/test"):
        calibrate_support_threshold(rows)


@pytest.mark.slow
def test_official_minicheck_integration_when_model_available():
    if os.getenv("RIVER_RUN_MINICHECK_SLOW") != "1":
        pytest.skip("set RIVER_RUN_MINICHECK_SLOW=1 when official MiniCheck weights are available")
    text = "A group of students gather in the school library to study for final exams."
    target = claim("c1", "The students are preparing for an examination.", ("doc", "chunk-a"))
    exact = ExactChunkVerifier(MiniCheckVerifier())
    results = exact.verify_claims([target], [chunk("doc", "chunk-a", text)])

    assert results[0].label == VerificationLabel.SUPPORTED
