"""Release-critical tests for strict direct-answer scoring."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCORER = Path(__file__).resolve().parents[1] / "scripts" / "release" / "score_answer_text.py"
spec = importlib.util.spec_from_file_location("score_answer_text", SCORER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def item(gold="h_gold", *, gold_object="The Daily Worker", other_object="The Hoover Gazette"):
    return {
        "gold_state": "active",
        "gold_hypothesis_id": gold,
        "transformation_metadata": {
            "gold_answer": gold_object,
            "gold_answer_aliases": [gold_object.replace("The ", "")],
        },
        "hypotheses": [
            {"hypothesis_id": "h_gold", "object": gold_object},
            {"hypothesis_id": "h_alt", "object": other_object},
        ],
    }


def test_exact_final_answer_is_correct():
    assert module.score(item(), "FINAL_ANSWER: The Daily Worker\nExplanation: supported [c1].") is True


def test_recorded_alias_is_correct():
    assert module.score(item(), "FINAL_ANSWER: Daily Worker") is True


def test_wrong_first_answer_is_not_rescued_by_later_gold_mention():
    answer = "FINAL_ANSWER: The Hoover Gazette\nThe Daily Worker is mentioned in another source."
    assert module.score(item(), answer) is False


def test_plain_prose_without_contract_is_wrong():
    assert module.score(item(), "The answer is The Daily Worker.") is False
    assert module.parse_final_answer("The answer is The Daily Worker.") is None


def test_only_first_nonempty_line_can_be_the_direct_answer():
    answer = "The Hoover Gazette is likely.\nFINAL_ANSWER: The Daily Worker"
    assert module.score(item(), answer) is False


def test_typography_is_normalised_conservatively():
    target = item(gold_object="O’Brien’s law")
    assert module.score(target, "FINAL_ANSWER: **O'Brien's law**.") is True


def test_rows_without_one_gold_answer_are_not_scored():
    disputed = item(gold=None)
    disputed["gold_state"] = "disputed"
    assert module.score(disputed, "DISPUTED: sources conflict.") is None


def test_named_hypothesis_uses_only_direct_answer_field():
    target = item()
    answer = "FINAL_ANSWER: The Hoover Gazette\nThe Daily Worker appears later."
    direct = module.parse_final_answer(answer)
    assert module.named_hypothesis(target, direct) == "h_alt"


def test_decision_fidelity_detects_generator_override():
    target = item()
    row = {"predicted_state": "active", "predicted_hypothesis_id": "h_alt"}
    assert module.decision_faithful(target, row, "The Daily Worker") is False


def test_non_active_contract_uses_fixed_state_prefix():
    row = {"predicted_state": "abstain"}
    assert module.contract_valid(row, "ABSTAIN: insufficient supported evidence.") is True
    assert module.contract_valid(row, "FINAL_ANSWER: The Daily Worker") is False
