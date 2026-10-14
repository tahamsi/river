"""JSON-stdin runner for the Microsoft GraphRAG native RIVER baseline.

Uses the pinned upstream checkout (``external/graphrag``, commit 14a00ad88fc3)
through ``graphrag.api``. The turn's evidence is written as an input corpus, the
GraphRAG pipeline builds its own index over it, and the question is answered by
GraphRAG's local or global search. Retrieval, chunking, entity extraction and
community reporting are GraphRAG's own, which is what makes this the native
comparison mode rather than a controlled one.

Indexing is expensive, so a session's index is built once and reused: the
workspace is keyed by ``session_id`` under ``GRAPHRAG_WORKSPACE`` and rebuilt
only when the evidence bundle changes.

GraphRAG returns a narrative response and emits no revision state of its own, so
this runner reports ``active`` when search produced a response and ``abstain``
when it produced none. It never emits ``disputed``, ``evolution`` or
``unknown``; the adapter records those as unsupported for this system.

Note that ``graphrag`` at this pin is a monorepo whose root package installs no
modules. The importable distribution is ``packages/graphrag``; see
``scripts/serving/setup_native_baselines.sh``.

Environment:
    GRAPHRAG_WORKSPACE       index root, default artifacts/graphrag
    GRAPHRAG_SEARCH_METHOD   "local" (default) or "global"
    GRAPHRAG_COMMUNITY_LEVEL default 2
    GRAPHRAG_RESPONSE_TYPE   default "Multiple Paragraphs"
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

import river_runner_contract as contract

SEARCH_LOCAL = "local"
SEARCH_GLOBAL = "global"

# Tables produced by the indexing pipeline that the search entry points consume.
LOCAL_TABLES = ("entities", "communities", "community_reports", "text_units", "relationships")
GLOBAL_TABLES = ("entities", "communities", "community_reports")


def _shared_corpus() -> list[str] | None:
    """Every evidence sentence in the fixture named by GRAPHRAG_SHARED_CORPUS.

    Returns None when unset, which keeps the per-request behaviour that the
    single-example verification exercises.
    """

    location = os.environ.get("GRAPHRAG_SHARED_CORPUS")
    if not location:
        return None
    import json as _json

    fixture = Path(location)
    if not fixture.exists():
        raise contract.RunnerError(f"GRAPHRAG_SHARED_CORPUS does not exist: {fixture}")
    seen: dict[str, None] = {}
    for line in fixture.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for evidence in _json.loads(line).get("evidence") or []:
            text_value = str((evidence or {}).get("text") or "").strip()
            if text_value:
                seen.setdefault(text_value, None)
    if not seen:
        raise contract.RunnerError(f"no evidence text in {fixture}")
    return list(seen)


def _embedding_dimension(settings: dict[str, str]) -> int:
    """Width of one embedding from the configured endpoint.

    Asked rather than assumed. The vector store's own default is 3072, which is
    correct for text-embedding-3-large and wrong for every local model this
    project runs, and the mismatch does not surface until a query reaches lance.
    """

    import urllib.error
    import urllib.request
    import json as _json

    request = urllib.request.Request(
        settings["embedding_base_url"].rstrip("/") + "/embeddings",
        data=_json.dumps({"model": settings["embedding_model"], "input": "dimension probe"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {settings['api_key']}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = _json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise contract.RunnerError(
            f"could not measure embedding width for {settings['embedding_model']}: {error}"
        ) from error
    try:
        return len(payload["data"][0]["embedding"])
    except (KeyError, IndexError, TypeError) as error:
        raise contract.RunnerError(
            f"embedding endpoint returned no vector for {settings['embedding_model']}"
        ) from error


def _workspace(payload: dict[str, Any], documents: list[str]) -> Path:
    """Per-session index directory, invalidated when the evidence changes."""

    # Absolute: graphrag's load_config changes the working directory, so a
    # relative workspace path stops resolving once configuration is loaded.
    root = Path(os.environ.get("GRAPHRAG_WORKSPACE", "artifacts/graphrag")).resolve()
    # A shared corpus is indexed once for every request, so the session must not
    # enter the key or each example would rebuild it.
    session = "shared" if os.environ.get("GRAPHRAG_SHARED_CORPUS") else str(payload.get("session_id", "smoke"))
    # The embedding model is part of the key: an index embedded at one
    # dimensionality cannot be queried at another, and lance reports that only
    # at search time as a dimension mismatch.
    settings = contract.local_llm()
    material = "\n".join(
        [*documents, settings["model"], settings["embedding_model"], str(_embedding_dimension(settings))]
    )
    fingerprint = hashlib.sha256(material.encode()).hexdigest()[:16]
    safe_session = "".join(char if char.isalnum() or char in "-_" else "_" for char in session)
    return root / f"{safe_session}-{fingerprint}"


def _write_corpus(workspace: Path, documents: list[str]) -> None:
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for index, text in enumerate(documents):
        (input_dir / f"evidence_{index:03d}.txt").write_text(text, encoding="utf-8")


def _point_at_local_endpoint(settings_path: Path, settings: dict[str, str]) -> None:
    """Rewrite the scaffolded model entries to use loopback inference.

    ``graphrag init`` writes an OpenAI-provider config with no ``api_base``,
    which would send requests off-box. RIVER's experiment scope is local
    open-weight inference only, so every model entry is repointed here.
    """

    import yaml

    config = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    for section, model_key in (
        ("completion_models", "model"),
        ("embedding_models", "model"),
    ):
        for entry in (config.get(section) or {}).values():
            entry["model_provider"] = "openai"
            entry.setdefault("auth_method", "api_key")
            # Defaults here are tuned for metered hosted APIs. The endpoint is
            # on loopback, so the only sensible limit is what the GPU sustains.
            entry["concurrent_requests"] = int(
                os.environ.get("GRAPHRAG_CONCURRENT_REQUESTS", "32")
            )
            entry["async_mode"] = "threaded"
            entry["requests_per_minute"] = None
            entry["tokens_per_minute"] = None
            if section == "embedding_models":
                entry["api_base"] = settings["embedding_base_url"]
                entry[model_key] = settings["embedding_model"]
            else:
                entry["api_base"] = settings["base_url"]
                entry[model_key] = settings["model"]
    # Without this the store is built at the library default of 3072 while the
    # configured model returns something else, and lance only says so at search
    # time.
    vector_store = config.setdefault("vector_store", {})
    vector_store["vector_size"] = _embedding_dimension(settings)
    settings_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _config(workspace: Path):
    """Scaffold the project if needed, then load its configuration."""

    from graphrag.cli.initialize import initialize_project_at
    from graphrag.config.load_config import load_config

    settings = contract.local_llm()
    # settings.yaml interpolates ${GRAPHRAG_API_KEY}; the local endpoint accepts
    # any non-empty value.
    os.environ.setdefault("GRAPHRAG_API_KEY", settings["api_key"])

    settings_path = workspace / "settings.yaml"
    if not settings_path.exists():
        initialize_project_at(
            path=workspace,
            force=False,
            model=settings["model"],
            embedding_model=settings["embedding_model"],
        )
        _point_at_local_endpoint(settings_path, settings)
    return load_config(workspace)


def _index_is_built(workspace: Path, tables: tuple[str, ...]) -> bool:
    output = workspace / "output"
    return all((output / f"{name}.parquet").exists() for name in tables)


def _load_tables(workspace: Path, tables: tuple[str, ...]) -> dict[str, Any]:
    import pandas as pd

    return {name: pd.read_parquet(workspace / "output" / f"{name}.parquet") for name in tables}


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    from graphrag import api

    query = str(payload.get("query", "")).strip()
    documents = contract.evidence_text(payload)
    if not query:
        return contract.abstained("empty query", native_state="no_query")
    if not documents:
        return contract.abstained("no documents to index", native_state="empty_corpus")

    method = os.environ.get("GRAPHRAG_SEARCH_METHOD", SEARCH_LOCAL).lower()
    if method not in {SEARCH_LOCAL, SEARCH_GLOBAL}:
        raise contract.RunnerError(
            f"GRAPHRAG_SEARCH_METHOD must be {SEARCH_LOCAL!r} or {SEARCH_GLOBAL!r}, got {method!r}"
        )
    tables = LOCAL_TABLES if method == SEARCH_LOCAL else GLOBAL_TABLES

    # Indexing the whole corpus once is how GraphRAG is meant to run, and the
    # only way it finishes: a per-request index costs about thirteen minutes.
    corpus = _shared_corpus() or documents
    workspace = _workspace(payload, corpus)
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / "input").exists():
        _write_corpus(workspace, corpus)
    config = _config(workspace)

    if not _index_is_built(workspace, tables):
        results = asyncio.run(api.build_index(config=config))
        failures = [result for result in results if getattr(result, "errors", None)]
        if failures:
            raise contract.RunnerError(
                "graphrag indexing failed: "
                + "; ".join(str(result.errors) for result in failures[:3])
            )

    frames = _load_tables(workspace, tables)
    community_level = int(os.environ.get("GRAPHRAG_COMMUNITY_LEVEL", "2"))
    response_type = os.environ.get("GRAPHRAG_RESPONSE_TYPE", "Multiple Paragraphs")

    if method == SEARCH_LOCAL:
        response, _context = asyncio.run(
            api.local_search(
                config=config,
                entities=frames["entities"],
                communities=frames["communities"],
                community_reports=frames["community_reports"],
                text_units=frames["text_units"],
                relationships=frames["relationships"],
                covariates=None,
                community_level=community_level,
                response_type=response_type,
                query=query,
            )
        )
    else:
        response, _context = asyncio.run(
            api.global_search(
                config=config,
                entities=frames["entities"],
                communities=frames["communities"],
                community_reports=frames["community_reports"],
                community_level=community_level,
                dynamic_community_selection=False,
                response_type=response_type,
                query=query,
            )
        )

    answer = response if isinstance(response, str) else str(response)
    if not answer.strip():
        return contract.abstained("search returned no response", native_state=f"{method}_search_empty")

    return contract.result(
        answer.strip(),
        predicted_state="active",
        citation_supported=None,  # GraphRAG cites communities, not RIVER chunk ids
        native_state=f"{method}_search",
    )


if __name__ == "__main__":
    contract.run(handle)
