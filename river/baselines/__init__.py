"""Baseline implementations and native-system adapters."""

from .base import Baseline, BaselineCapabilities, BaselineUnavailable
from .controls import LastWriteWins, NaiveRAG, RecencyWeightedMemory, StatelessRAG
from .external import (
    CRAGAdapter, GraphRAGAdapter, GraphRAGGlobalAdapter, GraphRAGLocalAdapter, MemGPTAdapter,
    SelfRAGAdapter, ZepAdapter,
)
from .minicheck import MiniCheckNoRevisionBaseline
from .river import RIVERDBaseline, RIVERLBaseline

__all__ = [
    "Baseline", "BaselineCapabilities", "BaselineUnavailable", "NaiveRAG", "StatelessRAG",
    "LastWriteWins", "RecencyWeightedMemory",
    "RIVERDBaseline", "RIVERLBaseline", "SelfRAGAdapter", "CRAGAdapter", "GraphRAGAdapter",
    "GraphRAGLocalAdapter", "GraphRAGGlobalAdapter", "MemGPTAdapter", "ZepAdapter",
    "MiniCheckNoRevisionBaseline",
]
