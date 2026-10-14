"""JSON-stdin runner for the Self-RAG native RIVER baseline.

Runs the released ``selfrag/selfrag_llama2_7b`` checkpoint through vLLM using the
prompt format and reflection-token vocabulary of the pinned upstream checkout
(``external/self-rag``, commit 1fcdc420e48f). Self-RAG emits its own critique
tokens; this runner maps them onto RIVER states and preserves the raw token so
the mapping stays auditable.

Reflection tokens, from ``retrieval_lm/utils.py`` upstream:

* relevance  - ``[Relevant]`` / ``[Irrelevant]``
* grounding  - ``[Fully supported]`` / ``[Partially supported]``
                / ``[No support / Contradictory]``
* utility    - ``[Utility:1]`` .. ``[Utility:5]``

Mapping to RIVER states:

* ``[Irrelevant]`` or ``[No support / Contradictory]`` -> ``abstain``
* otherwise                                            -> ``active``

Self-RAG expresses no notion of concurrent dispute or temporal evolution, so
this runner never emits ``disputed`` or ``evolution``. The adapter records those
states as unsupported for this system rather than evaluator-derived.

Environment:
    SELF_RAG_MODEL          model id or local path (default selfrag/selfrag_llama2_7b)
    SELF_RAG_DOWNLOAD_DIR   vLLM download directory (default artifacts/hf_cache)
    SELF_RAG_MAX_TOKENS     generation cap (default 200)
    SELF_RAG_DTYPE          vLLM dtype (default half)
    SELF_RAG_GPU_FRACTION   vLLM gpu_memory_utilization (default 0.85)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import river_runner_contract as contract

RELEVANCE_IRRELEVANT = "[Irrelevant]"
GROUND_NO_SUPPORT = "[No support / Contradictory]"
GROUND_TOKENS = ("[Fully supported]", "[Partially supported]", GROUND_NO_SUPPORT)
CONTROL_TOKENS = (
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
    "[No Retrieval]", "[Retrieval]", "[Irrelevant]", "[Relevant]",
    "<paragraph>", "</paragraph>",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
)

# The instruction format the released checkpoint was trained against.
PROMPT = (
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


@lru_cache(maxsize=1)
def _model():
    """Load the Self-RAG checkpoint once per process."""

    from vllm import LLM

    return LLM(
        model=os.environ.get("SELF_RAG_MODEL", "selfrag/selfrag_llama2_7b"),
        download_dir=os.environ.get("SELF_RAG_DOWNLOAD_DIR", "artifacts/hf_cache"),
        dtype=os.environ.get("SELF_RAG_DTYPE", "half"),
        gpu_memory_utilization=float(os.environ.get("SELF_RAG_GPU_FRACTION", "0.85")),
    )


def _sampling_params():
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(os.environ.get("SELF_RAG_MAX_TOKENS", "200")),
        skip_special_tokens=False,  # reflection tokens are the signal
    )


def _prompt(query: str, passages: list[str]) -> str:
    """Upstream short-form prompt with each chunk wrapped as a paragraph."""

    instruction = query
    if passages:
        joined = "\n".join(f"<paragraph>{text}</paragraph>" for text in passages)
        instruction = f"{query}\n\n{joined}"
    return PROMPT.format(instruction=instruction)


def _clean(text: str) -> str:
    for token in CONTROL_TOKENS:
        text = text.replace(token, " ")
    return " ".join(text.split())


def _critique(text: str) -> tuple[str | None, str | None]:
    """Extract the relevance and grounding tokens Self-RAG emitted."""

    relevance = RELEVANCE_IRRELEVANT if RELEVANCE_IRRELEVANT in text else (
        "[Relevant]" if "[Relevant]" in text else None
    )
    grounding = next((token for token in GROUND_TOKENS if token in text), None)
    return relevance, grounding


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    passages = contract.evidence_text(payload)
    query = str(payload.get("query", "")).strip()
    if not query:
        return contract.abstained("empty query", native_state="no_query")

    outputs = _model().generate([_prompt(query, passages)], _sampling_params())
    completion = outputs[0].outputs[0]
    raw = completion.text
    relevance, grounding = _critique(raw)
    native_state = " ".join(token for token in (relevance, grounding) if token) or "no_reflection_token"

    if relevance == RELEVANCE_IRRELEVANT or grounding == GROUND_NO_SUPPORT:
        return contract.abstained("self_rag_declined", native_state=native_state)

    return contract.result(
        _clean(raw),
        predicted_state="active",
        citation_supported=grounding in {"[Fully supported]", "[Partially supported]"},
        input_tokens=len(outputs[0].prompt_token_ids),
        output_tokens=len(completion.token_ids),
        native_state=native_state,
    )


if __name__ == "__main__":
    contract.run(handle)
