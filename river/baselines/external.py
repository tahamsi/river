"""Compatibility imports for native external baseline adapters."""

from .native import (
    CRAGAdapter,
    GraphRAGAdapter,
    GraphRAGGlobalAdapter,
    GraphRAGLocalAdapter,
    MemGPTAdapter,
    SelfRAGAdapter,
    ZepAdapter,
)

__all__ = [
    "CRAGAdapter",
    "GraphRAGAdapter",
    "GraphRAGGlobalAdapter",
    "GraphRAGLocalAdapter",
    "MemGPTAdapter",
    "SelfRAGAdapter",
    "ZepAdapter",
]
