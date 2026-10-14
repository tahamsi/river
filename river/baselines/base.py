from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


class BaselineUnavailable(RuntimeError):
    """Raised when a native external baseline has not been configured."""


@dataclass(frozen=True)
class BaselineCapabilities:
    persistent_memory: bool = False
    temporal_validity: bool = False
    citation_verification: bool = False
    conflict_resolution: bool = False
    abstention: bool = False
    native_external_system: bool = False


class Baseline(ABC):
    name = "baseline"
    version = "1"
    capabilities = BaselineCapabilities()

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def reset(self, session_id: str) -> None:
        return None

    @abstractmethod
    def process_turn(self, session_id: str, query: str, retrieved_chunks: list, target_time=None):
        raise NotImplementedError

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": asdict(self.capabilities),
            "config": self.config,
        }


def chunk_text(chunks: list) -> str:
    return "\n\n".join(
        item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "") for item in chunks
    )
