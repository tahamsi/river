import pytest

from river.evaluation.metrics import EvaluationError, validate_no_calibration_test_mix, validate_single_comparison_mode
from river.evaluation.statistics import (
    benjamini_hochberg_correction,
    effect_sizes,
    holm_correction,
    mcnemar_test,
    non_inferiority_test,
    paired_bootstrap_ci,
    paired_permutation_test,
    seed_aggregation,
)


def test_bootstrap_and_effect_size_use_paired_differences():
    left = [1.0, 0.0, 1.0]
    right = [0.0, 0.0, 1.0]

    ci = paired_bootstrap_ci(left, right, n_resamples=200, seed=1)
    effects = effect_sizes(left, right)

    assert ci["estimate"] == pytest.approx(1 / 3)
    assert ci["n"] == 3
    assert effects["mean_difference"] == pytest.approx(1 / 3)
    assert effects["n"] == 3


def test_mcnemar_and_permutation_tests_are_hand_calculated():
    left = [True, True, False, False]
    right = [True, False, True, False]

    mcnemar = mcnemar_test(left, right)
    permutation = paired_permutation_test([1, 0], [0, 0], n_permutations=10)

    assert mcnemar["b_left_only"] == 1
    assert mcnemar["c_right_only"] == 1
    assert mcnemar["p_value"] == 1.0
    assert permutation["estimate"] == 0.5
    assert permutation["n"] == 2


def test_p_value_corrections():
    p_values = {"a": 0.01, "b": 0.03, "c": 0.2}

    assert holm_correction(p_values)["a"] == pytest.approx(0.03)
    assert benjamini_hochberg_correction(p_values)["b"] == pytest.approx(0.045)


def test_non_inferiority_uses_candidate_minus_reference_margin():
    result = non_inferiority_test([1, 1, 1, 0], [1, 1, 0, 0], margin=0.1)

    assert result["observed_difference"] == pytest.approx(0.25)
    assert result["margin"] == 0.1
    assert result["n"] == 4


def test_seed_aggregation_keeps_seed_units():
    records = [
        {"baseline": "a", "comparison_mode": "controlled", "seed": 1, "intervention": "supported_genuine_update", "gold_state": "active", "predicted_state": "active", "gold_hypothesis_id": "h", "predicted_hypothesis_id": "h", "intervention_hypothesis_id": "h"},
        {"baseline": "a", "comparison_mode": "controlled", "seed": 2, "intervention": "supported_genuine_update", "gold_state": "active", "predicted_state": "active", "gold_hypothesis_id": "h", "predicted_hypothesis_id": "x", "intervention_hypothesis_id": "h"},
    ]

    rows = seed_aggregation(records, ["genuine_update_accuracy"])

    assert rows[0]["n_seeds"] == 2
    assert rows[0]["genuine_update_accuracy_mean"] == 0.5


def test_validation_refuses_mixed_modes_and_calibration_test_splits():
    with pytest.raises(EvaluationError, match="controlled/native"):
        validate_single_comparison_mode([
            {"comparison_mode": "controlled"},
            {"comparison_mode": "native"},
        ])

    with pytest.raises(EvaluationError, match="mixed calibration"):
        validate_no_calibration_test_mix([
            {"split": "validation"},
            {"split": "test"},
        ])
