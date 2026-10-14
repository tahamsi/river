"""JSON-stdin runner wrapper for the Zep/Graphiti native RIVER baseline.

Reads RIVER's runner_command contract on stdin (see river/baselines/native.py
NativeCommandAdapter.process_turn) and writes back a JSON object with at
least `answer`; `predicted_state` is one of active/disputed/evolution/
unknown/abstain per river/schemas.RevisionState.

This ingests the turn's evidence chunks into a Graphiti temporal graph
(scoped to the request's session_id as the group_id, so repeated smoke runs
do not bleed into each other) and answers by searching that graph for facts
relevant to the query. It uses the local Ollama OpenAI-compatible endpoint
for both the LLM and the embedder, per RIVER's local-only-inference rule.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


async def _run(payload: dict) -> dict:
    base_url = _env("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.environ.get("LOCAL_LLM_API_KEY", "ollama")
    model = os.environ.get("LOCAL_LLM_MODEL", "qwen3.5:4b")
    embedding_model = os.environ.get("LOCAL_EMBEDDING_MODEL", "qwen3-embedding:0.6b")

    llm_config = LLMConfig(api_key=api_key, model=model, base_url=base_url)
    llm_client = OpenAIClient(config=llm_config)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(api_key=api_key, embedding_model=embedding_model, base_url=base_url)
    )

    graphiti = Graphiti(
        uri=_env("NEO4J_URI"),
        user=_env("NEO4J_USER"),
        password=_env("NEO4J_PASSWORD"),
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=OpenAIRerankerClient(config=llm_config),
    )

    group_id = f"river-{payload.get('session_id', 'smoke')}"
    reference_time = datetime.now(timezone.utc)

    try:
        # build_indices_and_constraints() is idempotent state, not per-call setup:
        # re-running it against an already-initialized graph makes Neo4j raise
        # EquivalentSchemaRuleAlreadyExists, and graphiti's driver prints that
        # failure to *stdout* (not stderr, not swallowed), which corrupts the
        # JSON-stdin contract's stdout channel. Only build them once.
        if os.environ.get("RIVER_ZEP_SKIP_INDEX_BUILD") != "1":
            await graphiti.build_indices_and_constraints()

        for index, item in enumerate(payload.get("evidence", [])):
            await graphiti.add_episode(
                name=f"{group_id}-evidence-{index}",
                episode_body=item.get("text", ""),
                source_description=item.get("source_id", "evidence"),
                reference_time=reference_time,
                source=EpisodeType.text,
                group_id=group_id,
            )

        edges = await graphiti.search(payload.get("query", ""), group_ids=[group_id], num_results=5)

        if not edges:
            return {
                "answer": "",
                "predicted_state": "abstain",
                "citation_supported": False,
                "input_tokens": None,
                "output_tokens": None,
            }

        facts = [edge.fact for edge in edges if getattr(edge, "fact", None)]
        return {
            "answer": " ".join(facts),
            "predicted_state": "active",
            "citation_supported": True,
            "input_tokens": None,
            "output_tokens": None,
        }
    finally:
        await graphiti.close()


def main() -> None:
    payload = json.loads(sys.stdin.read())
    result = asyncio.run(_run(payload))
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
