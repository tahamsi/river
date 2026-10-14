from types import SimpleNamespace

import pytest

from river.generation import LocalOpenAICompatibleGenerator
from experiments.run import build_local_generator, generation_context, structured_answer_correct


def test_ollama_generator_id_enables_local_model():
    args = SimpleNamespace(
        local_model=None,
        generator="ollama/qwen3.5:4b",
        local_base_url="http://127.0.0.1:11434/v1",
        temperature=0.0,
        max_output_tokens=64,
        model_timeout=5.0,
        seed=42,
    )
    generator = build_local_generator(args)
    assert generator.model == "qwen3.5:4b"
    assert generator.manifest()["remote_paid_api"] is False


def test_generator_rejects_remote_or_deceptive_endpoint():
    with pytest.raises(ValueError, match="localhost"):
        LocalOpenAICompatibleGenerator("model", base_url="https://hosted-model.example/v1")
    with pytest.raises(ValueError, match="localhost"):
        LocalOpenAICompatibleGenerator("model", base_url="http://localhost.example.com/v1")


def test_generation_context_limits_active_decision_to_selected_hypothesis():
    item = {
        "hypotheses": [
            {"hypothesis_id": "old", "subject": "A", "predicate": "is", "object": "old"},
            {"hypothesis_id": "new", "subject": "A", "predicate": "is", "object": "new"},
        ],
        "evidence": [{"chunk_id": "c1", "text": "A is new."}],
    }
    context = generation_context(item, {"predicted_state": "active", "predicted_hypothesis_id": "new"})
    assert "Selected answer: new" in context
    assert "Selected answer: old" not in context
    assert "[c1]" in context


def test_structured_correctness_uses_state_and_selected_hypothesis():
    item = {"gold_state": "active", "gold_hypothesis_id": "new"}
    assert structured_answer_correct(item, {"predicted_state": "active", "predicted_hypothesis_id": "new"}) is True
    assert structured_answer_correct(item, {"predicted_state": "active", "predicted_hypothesis_id": "old"}) is False
    assert structured_answer_correct({"gold_state": "disputed"}, {"predicted_state": "disputed"}) is True
