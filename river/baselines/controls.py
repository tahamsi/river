from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

from .base import Baseline, BaselineCapabilities, chunk_text


def _default_generator(query: str, context: str) -> str:
    return context[:1000] if context else "I do not have sufficient evidence."


class NaiveRAG(Baseline):
    name = "naive_rag"

    def __init__(self, config=None, generator: Callable[[str, str], str] | None = None):
        super().__init__(config)
        self.generator = generator or _default_generator

    def process_turn(self, session_id, query, retrieved_chunks, target_time=None):
        return {"answer": self.generator(query, chunk_text(retrieved_chunks)), "predicted_state": None}


class StatelessRAG(NaiveRAG):
    name = "stateless_rag"


class LastWriteWins(Baseline):
    name = "last_write_wins"
    capabilities = BaselineCapabilities(persistent_memory=True, conflict_resolution=True)

    def __init__(self, config=None):
        super().__init__(config)
        self.sessions: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)

    def reset(self, session_id):
        self.sessions.pop(session_id, None)

    def process_turn(self, session_id, query, retrieved_chunks, target_time=None):
        for item in retrieved_chunks:
            if isinstance(item, dict) and {"subject", "predicate", "object"} <= item.keys():
                self.sessions[session_id][(item["subject"], item["predicate"])] = item
        values = list(self.sessions[session_id].values())
        selected = values[-1] if values else None
        return {
            "answer": selected.get("object", "") if selected else "",
            "predicted_state": "active" if selected else "abstain",
            "predicted_hypothesis_id": selected.get("hypothesis_id") if selected else None,
        }


class RecencyWeightedMemory(LastWriteWins):
    name = "recency_weighted"

    def __init__(self, config=None):
        super().__init__(config)
        self.memories: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def process_turn(self, session_id, query, retrieved_chunks, target_time=None):
        self.memories[session_id].extend(item for item in retrieved_chunks if isinstance(item, dict))
        gamma = float(self.config.get("recency_weight", 1.0))
        decay = float(self.config.get("recency_decay", 0.995))
        relevance_weight = float(self.config.get("relevance_weight", 1.0))
        ranked = []
        total = len(self.memories[session_id])
        for index, item in enumerate(self.memories[session_id]):
            age = total - index - 1
            score = relevance_weight * float(item.get("retrieval_score", 0.0)) + gamma * math.pow(decay, age)
            ranked.append((score, index, item))
        selected = max(ranked, default=(0, -1, None), key=lambda row: (row[0], row[1]))[2]
        return {
            "answer": selected.get("object", selected.get("text", "")) if selected else "",
            "predicted_state": "active" if selected else "abstain",
            "predicted_hypothesis_id": selected.get("hypothesis_id") if selected else None,
        }
