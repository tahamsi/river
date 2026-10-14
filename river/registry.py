from __future__ import annotations

from typing import Any

from .baselines import (
    CRAGAdapter, GraphRAGAdapter, GraphRAGGlobalAdapter, GraphRAGLocalAdapter, LastWriteWins,
    MemGPTAdapter, MiniCheckNoRevisionBaseline, NaiveRAG,
    RIVERDBaseline, RIVERLBaseline, RecencyWeightedMemory,
    SelfRAGAdapter, StatelessRAG, ZepAdapter,
)

BASELINE_REGISTRY = {
    "naive_rag": NaiveRAG,
    "stateless_rag": StatelessRAG,
    "last_write_wins": LastWriteWins,
    "recency_weighted": RecencyWeightedMemory,
    "minicheck_no_revision": MiniCheckNoRevisionBaseline,
    "self_rag": SelfRAGAdapter,
    "crag": CRAGAdapter,
    "graphrag": GraphRAGAdapter,
    "graphrag_local": GraphRAGLocalAdapter,
    "graphrag_global": GraphRAGGlobalAdapter,
    "memgpt": MemGPTAdapter,
    "zep": ZepAdapter,
    "river_d": RIVERDBaseline,
    "river_l": RIVERLBaseline,
}


def available_baselines() -> tuple[str, ...]:
    return tuple(sorted(BASELINE_REGISTRY))


def build_baseline(name: str, config: dict[str, Any] | None = None):
    try:
        baseline = BASELINE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown baseline {name!r}; choose from {available_baselines()}") from exc
    return baseline(config=config)
