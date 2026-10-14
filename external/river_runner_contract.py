"""Shared JSON-stdin contract for RIVER's native baseline runners.

Each native system runs in its own virtual environment under ``external/envs``
and cannot import the ``river`` package, so this module is standalone and uses
only the standard library. It sits beside the runners in ``external/`` and is
importable because Python puts a script's own directory on ``sys.path``.

The contract, as consumed by ``river.baselines.native.NativeCommandAdapter``:

* stdin  - one JSON object with ``baseline_id``, ``comparison_mode``,
  ``session_id``, ``query``, ``target_time``, ``evidence``, ``budgets`` and
  ``native_config``.
* stdout - one JSON object with at least ``answer``. ``predicted_state``, when
  emitted, must be one of the five ``river.schemas.RevisionState`` values.

A runner must not invent a state its system does not express. Leaving
``predicted_state`` unset is recorded by the adapter as unsupported; guessing
one would misreport the native system's behaviour.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any, Iterator

#: The five states in ``river.schemas.RevisionState``.
REVISION_STATES = frozenset({"active", "disputed", "evolution", "unknown", "abstain"})


class RunnerError(RuntimeError):
    """Raised for a setup or contract problem the operator must fix."""


def env(name: str, default: str | None = None) -> str:
    """Read a required environment variable, or fail with an actionable message."""

    value = os.environ.get(name, default)
    if value is None:
        raise RunnerError(
            f"missing required environment variable: {name}. "
            "Load infrastructure/native.env before invoking this runner."
        )
    return value


def local_llm() -> dict[str, str]:
    """Endpoint for the local open-weight model.

    RIVER's experiment scope is local inference only. The OpenAI-compatible
    protocol name does not imply a hosted model: this points at Ollama, vLLM or
    llama.cpp on the loopback interface.
    """

    return {
        "base_url": os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        # vLLM serves one model per process, so embeddings may sit on another
        # port. Falls back to the chat endpoint, which is where Ollama keeps it.
        "embedding_base_url": os.environ.get(
            "LOCAL_EMBEDDING_BASE_URL",
            os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        ),
        "api_key": os.environ.get("LOCAL_LLM_API_KEY", "ollama"),
        "model": os.environ.get("LOCAL_LLM_MODEL", "qwen3.5:4b"),
        "embedding_model": os.environ.get("LOCAL_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
    }


def read_payload() -> dict[str, Any]:
    """Read and parse the single JSON request object from stdin."""

    raw = sys.stdin.read()
    if not raw.strip():
        raise RunnerError("no JSON payload on stdin")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RunnerError(f"stdin was not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise RunnerError(f"expected a JSON object on stdin, got {type(payload).__name__}")
    return payload


def evidence_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Evidence chunks from the request, ignoring any non-object entries."""

    return [item for item in payload.get("evidence", []) if isinstance(item, dict)]


def evidence_text(payload: dict[str, Any]) -> list[str]:
    """Non-empty chunk texts, in the order the request presented them."""

    return [text for item in evidence_items(payload) if (text := str(item.get("text", "")).strip())]


def result(
    answer: str,
    *,
    predicted_state: str | None = None,
    citation_supported: bool | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    model_calls: int | None = None,
    native_state: str | None = None,
) -> dict[str, Any]:
    """Build a contract-shaped response.

    ``native_state`` carries the system's own label (Self-RAG's reflection token,
    CRAG's evaluator verdict) so the mapping to a RIVER state stays auditable
    rather than being silently collapsed.
    """

    if predicted_state is not None and predicted_state not in REVISION_STATES:
        raise RunnerError(
            f"{predicted_state!r} is not a RIVER revision state; expected one of {sorted(REVISION_STATES)}"
        )
    payload: dict[str, Any] = {
        "answer": answer,
        "predicted_state": predicted_state,
        "citation_supported": citation_supported,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if model_calls is not None:
        # Generator invocations for this turn. A native system that loops over
        # several calls is the only thing that can count them, because the
        # harness sees one subprocess and the endpoint may expose no counters.
        payload["model_calls"] = model_calls
    if native_state is not None:
        payload["native_state"] = native_state
    return payload


def abstained(reason: str, *, native_state: str | None = None) -> dict[str, Any]:
    """Response for a turn the system declined to answer."""

    return result(
        "",
        predicted_state="abstain",
        citation_supported=False,
        native_state=native_state or reason,
    )


@contextlib.contextmanager
def protected_stdout() -> Iterator[None]:
    """Keep library chatter off the channel carrying the JSON response.

    Several upstream systems print progress, warnings and driver errors to
    stdout, which corrupts the response and makes the adapter report a non-JSON
    runner instead of the real failure. Everything written inside this block is
    redirected to stderr, where the adapter surfaces it on error.
    """

    with contextlib.redirect_stdout(sys.stderr):
        yield


def run(handler) -> None:
    """Execute ``handler(payload)`` under the contract and emit its response.

    A setup or runtime failure exits non-zero with the reason on stderr, which
    is what ``NativeCommandAdapter`` turns into an actionable message rather
    than a silently degraded result.
    """

    try:
        payload = read_payload()
        with protected_stdout():
            response = handler(payload)
    except RunnerError as error:
        print(f"{error}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as error:  # noqa: BLE001 - reported verbatim to the adapter
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
    json.dump(response, sys.stdout)
    sys.stdout.flush()
