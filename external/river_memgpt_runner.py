"""JSON-stdin runner for the MemGPT / Letta native RIVER baseline.

Drives a Letta agent from the pinned upstream checkout (``external/letta``,
commit b76da9092518) through the ``letta_client`` SDK against a locally running
Letta server. The turn's evidence is written into the agent's archival memory
and the query is then asked of the agent, so retrieval is MemGPT's own
paged-memory mechanism rather than RIVER's retrieval.

The agent is scoped to the request's ``session_id`` so repeated runs do not
share memory state, matching the isolation the Zep runner uses.

MemGPT emits no revision state of its own: it answers or it does not. This
runner therefore reports only ``active`` (the agent produced an assistant
message) or ``abstain`` (it produced none), and never ``disputed``,
``evolution`` or ``unknown``. The adapter records those as unsupported for this
system rather than inventing them.

Environment:
    LETTA_BASE_URL        Letta server, default http://127.0.0.1:8283
    LETTA_TOKEN           bearer token when the server requires one
    LETTA_MODEL           agent LLM handle, default from LOCAL_LLM_MODEL
    LETTA_EMBEDDING       embedding handle, default from LOCAL_EMBEDDING_MODEL
    LETTA_KEEP_AGENT      set to 1 to leave the agent in place for inspection
"""

from __future__ import annotations

import os
from typing import Any

import river_runner_contract as contract


def _client():
    from letta_client import Letta

    token = os.environ.get("LETTA_TOKEN")
    base_url = os.environ.get("LETTA_BASE_URL", "http://127.0.0.1:8283")
    return Letta(base_url=base_url, token=token) if token else Letta(base_url=base_url)


def _model_handles() -> tuple[str, str]:
    settings = contract.local_llm()
    return (
        os.environ.get("LETTA_MODEL", f"ollama/{settings['model']}"),
        os.environ.get("LETTA_EMBEDDING", f"ollama/{settings['embedding_model']}"),
    )


def _agent_name(session_id: Any) -> str:
    """Name the agent after the session, in the character set Letta accepts.

    Example ids carry colons, which Letta rejects because it names export files
    after the agent. Replacing them keeps one agent per example, which is what
    isolates the memory state between runs.
    """

    safe = "".join(
        character if character.isalnum() or character in " -_'" else "-"
        for character in str(session_id)
    )
    return f"river-{safe}"


def _assistant_text(response: Any) -> str:
    """Concatenate the assistant messages from a Letta response."""

    parts: list[str] = []
    for message in getattr(response, "messages", []) or []:
        if getattr(message, "message_type", None) != "assistant_message":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(getattr(block, "text", "")) for block in content if getattr(block, "text", None)
            )
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query", "")).strip()
    if not query:
        return contract.abstained("empty query", native_state="no_query")

    client = _client()
    model, embedding = _model_handles()
    agent = client.agents.create(
        name=_agent_name(payload.get("session_id", "smoke")),
        model=model,
        embedding=embedding,
        memory_blocks=[
            {
                "label": "persona",
                "value": (
                    "You answer strictly from your archival memory. If the memory does "
                    "not contain the answer, say that you do not have enough information."
                ),
            }
        ],
    )
    try:
        for item in contract.evidence_items(payload):
            text = str(item.get("text", "")).strip()
            if text:
                client.agents.passages.create(agent_id=agent.id, text=text)

        response = client.agents.messages.create(
            agent_id=agent.id,
            messages=[{"role": "user", "content": query}],
        )
        answer = _assistant_text(response)
        if not answer:
            return contract.abstained("agent produced no assistant message", native_state="no_reply")

        usage = getattr(response, "usage", None)
        return contract.result(
            answer,
            predicted_state="active",
            citation_supported=None,  # MemGPT emits no citations to verify
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            # One MemGPT turn is an agent loop, so the number of generator calls
            # is the number of steps it took, not one.
            model_calls=getattr(usage, "step_count", None),
            native_state="assistant_message",
        )
    finally:
        if os.environ.get("LETTA_KEEP_AGENT") != "1":
            try:
                client.agents.delete(agent_id=agent.id)
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                pass


if __name__ == "__main__":
    contract.run(handle)
