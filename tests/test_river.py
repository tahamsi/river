from datetime import datetime, timezone

from river.revision import RIVERDeterministic
from river.schemas import BeliefHypothesis, EvidenceRecord, RevisionState, ValidityInterval


NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def evidence(evidence_id, entailment, group, start=None, end=None):
    return EvidenceRecord(
        evidence_id=evidence_id, source_id=evidence_id, chunk_id=evidence_id, observed_at=NOW,
        valid_time=ValidityInterval(start=start, end=end), entailment=entailment,
        contradiction=0.0, neutral=1.0 - entailment, independence_group=group,
    )


def test_arrival_order_invariance():
    hypotheses = [
        BeliefHypothesis(hypothesis_id="h1", subject="x", predicate="is", object="a", evidence_ids=["e1"]),
        BeliefHypothesis(hypothesis_id="h2", subject="x", predicate="is", object="b", evidence_ids=["e2"]),
    ]
    records = [evidence("e1", 0.9, "g1"), evidence("e2", 0.4, "g2")]
    river = RIVERDeterministic()
    assert river.resolve(hypotheses, records, NOW) == river.resolve(hypotheses, list(reversed(records)), NOW)


def test_missing_evidence_abstains():
    assert RIVERDeterministic().resolve([], [], NOW).state == RevisionState.ABSTAIN


def test_repetition_is_discounted():
    hypothesis = BeliefHypothesis(
        hypothesis_id="h1", subject="x", predicate="is", object="a", evidence_ids=["e1", "e2"]
    )
    records = [evidence("e1", 0.5, "same"), evidence("e2", 0.5, "same")]
    decision = RIVERDeterministic().resolve([hypothesis], records, NOW)
    assert decision.scores["h1"] < 1.0
