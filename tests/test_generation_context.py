"""Generation must render the operator decision, never re-adjudicate it."""

from __future__ import annotations

from experiments.run import generation_context, render_answer, retrieval_generation_context
from river.generation import GenerationResult


def _item():
    return {
        "query": "Which team does Gerun play for?",
        "target_time": "2024-03-01T00:00:00Z",
        "hypotheses": [
            {"hypothesis_id": "h_current", "subject": "Gerun", "predicate": "plays for", "object": "Sendai"},
            {"hypothesis_id": "h_expired", "subject": "Gerun", "predicate": "plays for", "object": "Betis"},
        ],
        "evidence": [
            {
                "evidence_id": "e_current",
                "chunk_id": "e_current:chunk0",
                "hypothesis_id": "h_current",
                "text": "Gerun plays for Sendai.",
            },
            {
                "evidence_id": "e_expired",
                "chunk_id": "e_expired:chunk0",
                "hypothesis_id": "h_expired",
                "text": "Gerun plays for Betis.",
            },
        ],
    }


class Generator:
    def __init__(self):
        self.calls = []

    def generate(self, query, context, *, instruction=None):
        self.calls.append((query, context, instruction))
        return GenerationResult("FINAL_ANSWER: Sendai", 10, 3)


def test_active_context_excludes_rejected_hypothesis_and_evidence():
    context = generation_context(
        _item(), {"predicted_state": "active", "predicted_hypothesis_id": "h_current"}
    )

    assert "Selected answer: Sendai" in context
    assert "Gerun plays for Sendai" in context
    assert "Betis" not in context
    assert "e_expired" not in context


def test_attribution_cannot_add_evidence_for_an_unselected_hypothesis():
    context = generation_context(
        _item(),
        {
            "predicted_state": "active",
            "predicted_hypothesis_id": "h_current",
            "revision": {"supporting_evidence_ids": ["e_expired"]},
        },
    )

    assert "Sendai" in context
    assert "Betis" not in context


def test_abstention_is_fixed_and_never_calls_generator():
    generator = Generator()
    answer, input_tokens, output_tokens, model_calls = render_answer(
        _item(), {"predicted_state": "abstain", "predicted_hypothesis_id": None}, "river_l", generator
    )

    assert answer.startswith("ABSTAIN:")
    assert (input_tokens, output_tokens, model_calls) == (0, 0, 0)
    assert generator.calls == []


def test_active_decision_calls_generator_with_selected_context():
    generator = Generator()
    answer, _, _, model_calls = render_answer(
        _item(), {"predicted_state": "active", "predicted_hypothesis_id": "h_current"}, "river_l", generator
    )

    assert answer == "FINAL_ANSWER: Sendai"
    assert model_calls == 1
    assert len(generator.calls) == 1
    assert "Betis" not in generator.calls[0][1]


def test_stateless_context_contains_evidence_but_not_benchmark_hypotheses():
    context = retrieval_generation_context(_item())

    assert "Gerun plays for Sendai" in context
    assert "Gerun plays for Betis" in context
    assert "Hypothesis" not in context
