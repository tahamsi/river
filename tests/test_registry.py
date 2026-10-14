import pytest

from river import available_baselines, build_baseline
from river.baselines import BaselineUnavailable


def test_required_baselines_registered():
    assert {
        "naive_rag", "stateless_rag", "last_write_wins", "recency_weighted",
        "minicheck_no_revision", "self_rag", "crag", "graphrag",
        "graphrag_local", "graphrag_global", "memgpt", "zep", "river_d", "river_l",
    } <= set(available_baselines())


def test_unknown_baseline_fails_clearly():
    with pytest.raises(ValueError):
        build_baseline("not_a_baseline")


def test_learned_river_requires_checkpoint():
    with pytest.raises(BaselineUnavailable):
        build_baseline("river_l")
