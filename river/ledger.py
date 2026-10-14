"""Non-destructive hypothesis and evidence ledger."""

from collections import defaultdict

from .schemas import BeliefHypothesis, EvidenceRecord


class HypothesisLedger:
    def __init__(self) -> None:
        self.hypotheses: dict[str, BeliefHypothesis] = {}
        self.evidence: dict[str, EvidenceRecord] = {}
        self.by_key: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add_hypothesis(self, hypothesis: BeliefHypothesis) -> None:
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        self.by_key[(hypothesis.subject.casefold(), hypothesis.predicate.casefold())].add(
            hypothesis.hypothesis_id
        )

    def add_evidence(self, hypothesis_id: str, evidence: EvidenceRecord) -> None:
        if hypothesis_id not in self.hypotheses:
            raise KeyError(f"Unknown hypothesis: {hypothesis_id}")
        self.evidence[evidence.evidence_id] = evidence
        hypothesis = self.hypotheses[hypothesis_id]
        if evidence.evidence_id not in hypothesis.evidence_ids:
            hypothesis.evidence_ids.append(evidence.evidence_id)

    def competitors(self, subject: str, predicate: str) -> list[BeliefHypothesis]:
        ids = sorted(self.by_key[(subject.casefold(), predicate.casefold())])
        return [self.hypotheses[item] for item in ids]

    def evidence_for(self, hypotheses: list[BeliefHypothesis]) -> list[EvidenceRecord]:
        ids = {evidence_id for hypothesis in hypotheses for evidence_id in hypothesis.evidence_ids}
        return [self.evidence[item] for item in sorted(ids) if item in self.evidence]
