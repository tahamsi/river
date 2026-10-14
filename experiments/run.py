"""Configuration-driven baseline runner for normalized RIVER-Bench JSONL fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import tracemalloc
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from river import build_baseline
from river.generation import LocalOpenAICompatibleGenerator
from river.schemas import PredictionRecord


LOCAL_RENDER_BASELINES = {
    "naive_rag", "stateless_rag", "last_write_wins", "recency_weighted", "river_d", "river_l"
}

UNSTRUCTURED_RENDER_BASELINES = {"naive_rag", "stateless_rag"}
GENERATION_CONTRACT_VERSION = "river-decision-render-v2"
DECISION_RENDER_INSTRUCTION = """Render the supplied decision without changing it.
The first non-empty line MUST be exactly `FINAL_ANSWER: <one direct answer>`.
Copy only the object shown after `Selected answer:` into that field, preserving
its wording. Do not put a sentence, explanation, citation, or alternatives on
the first line.
Use only the selected hypothesis and evidence supplied below. Do not propose an
alternative answer and do not use outside knowledge. An optional explanation
may follow on later lines. Cite supplied chunk IDs in square brackets."""
RETRIEVAL_RENDER_INSTRUCTION = """Answer only from the supplied retrieved evidence.
The first non-empty line MUST be exactly `FINAL_ANSWER: <one direct answer>`.
Put only the shortest answer entity or answer string after the colon; do not
put a sentence, explanation, citation, or alternatives on the first line. An
optional explanation may follow on later lines.
Cite supplied chunk IDs in square brackets."""
NON_ACTIVE_ANSWERS = {
    "abstain": "ABSTAIN: insufficient supported evidence.",
    "unknown": "UNKNOWN: the available evidence does not determine an answer.",
    "disputed": "DISPUTED: the available evidence conflicts and no single answer is selected.",
    "evolution": "EVOLUTION: the answer changes across valid time intervals.",
}


@contextmanager
def measure_revision():
    """Time and peak-allocation probe around one revision decision.

    Wall time excludes answer generation so the number reflects the revision
    operator rather than the language model behind it. Peak memory is the
    Python allocation high-water mark for the call, which is comparable across
    baselines in a way that process RSS is not.
    """

    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    baseline_peak = tracemalloc.get_traced_memory()[1]
    measurement: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield measurement
    finally:
        measurement["revision_latency_ms"] = (time.perf_counter() - start) * 1000
        peak = tracemalloc.get_traced_memory()[1]
        measurement["peak_memory_mb"] = max(peak - baseline_peak, 0) / (1024 * 1024)
        if started_tracing:
            tracemalloc.stop()


def resolve_nli_calls(baseline, result: dict) -> int | None:
    """Verifier invocations for this turn, or None when it cannot be known.

    A baseline that runs no verifier makes exactly zero calls, which is a real
    measurement. A baseline that runs one but reports no counter yields None so
    the metric stays missing instead of being recorded as zero.
    """

    for source_dict in (result, result.get("verification") or {}):
        if isinstance(source_dict, dict) and source_dict.get("nli_calls") is not None:
            return int(source_dict["nli_calls"])
    verifier = getattr(baseline, "verifier", None)
    if verifier is None:
        return 0
    count = getattr(verifier, "call_count", None)
    return int(count) if count is not None else None


def digest(value) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def git_commit() -> str:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    head = revision.stdout.strip() if revision.returncode == 0 else "uncommitted"
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    diff = subprocess.run(["git", "diff", "HEAD", "--"], capture_output=True, text=True, check=False).stdout
    if not status.strip():
        return head
    state_hash = hashlib.sha256((status + "\n" + diff).encode()).hexdigest()[:12]
    return f"{head}+dirty:{state_hash}"


def evidence_for_baseline(item: dict) -> list[dict]:
    """Translate RIVER-Bench transaction-time evidence into runner evidence."""

    rows = []
    for evidence in item.get("evidence", []):
        if not isinstance(evidence, dict):
            rows.append(evidence)
            continue
        normalized = dict(evidence)
        if "transaction_time" in normalized and "observed_at" not in normalized:
            normalized["observed_at"] = normalized["transaction_time"]
        rows.append(normalized)
    return rows


def build_local_generator(args):
    model = args.local_model
    if model is None and args.generator.startswith("ollama/"):
        model = args.generator.removeprefix("ollama/")
    if model is None:
        return None
    return LocalOpenAICompatibleGenerator(
        model,
        base_url=args.local_base_url,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        timeout=args.model_timeout,
        seed=args.seed,
    )


def _as_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _expired_at(evidence: dict, target_time) -> bool:
    """Whether this evidence describes a period that had ended by the target time."""

    target = _as_datetime(target_time)
    interval = evidence.get("valid_time") or {}
    if target is None or not isinstance(interval, dict):
        return False
    start, end = _as_datetime(interval.get("start")), _as_datetime(interval.get("end"))
    if start is None and end is None:
        return False
    return (start is not None and target < start) or (end is not None and target > end)


def _selected_hypothesis(item: dict, result: dict) -> dict | None:
    selected = result.get("predicted_hypothesis_id")
    if not selected:
        return None
    return next(
        (row for row in item.get("hypotheses", []) if row.get("hypothesis_id") == selected),
        None,
    )


def _selected_evidence(item: dict, result: dict) -> list[dict]:
    """Return only evidence compatible with the operator's selected hypothesis.

    Set-aside evidence is deliberately excluded. Showing it to the generator
    allowed the generator to become a second revision system and override an
    abstention or a wrong hypothesis selection. Operator attribution may narrow
    the selected evidence, but may never add evidence tagged to another
    hypothesis.
    """

    selected = result.get("predicted_hypothesis_id")
    if not selected:
        return []
    revision = result.get("revision")
    supporting_ids: set[str] | None = None
    if isinstance(revision, dict) and revision.get("supporting_evidence_ids"):
        supporting_ids = {str(value) for value in revision["supporting_evidence_ids"]}

    selected_rows: list[dict] = []
    for row in item.get("evidence", []):
        if not isinstance(row, dict):
            continue
        hypothesis_id = row.get("hypothesis_id")
        if hypothesis_id not in (None, selected):
            continue
        if supporting_ids is not None and row.get("evidence_id") not in supporting_ids:
            continue
        selected_rows.append(row)

    # Some baselines do not expose evidence attribution. In that case the
    # hypothesis tag is the only admissible fallback; never fall back to all
    # evidence because that reintroduces the leak this contract removes.
    if not selected_rows and supporting_ids is not None:
        selected_rows = [
            row
            for row in item.get("evidence", [])
            if isinstance(row, dict) and row.get("hypothesis_id") == selected
        ]
    return selected_rows


def generation_context(item: dict, result: dict) -> str:
    """Render an active structured decision without exposing rejected evidence."""

    state = str(result.get("predicted_state") or "").lower()
    hypothesis = _selected_hypothesis(item, result)
    if state != "active" or hypothesis is None:
        raise ValueError("decision generation requires an active state and one selected hypothesis")

    evidence = _selected_evidence(item, result)
    hypothesis_text = (
        f"Selected answer: {hypothesis.get('object')}\n"
        f"Selected proposition: {hypothesis.get('subject')} {hypothesis.get('predicate')} {hypothesis.get('object')}"
    )
    rendered_evidence = "\n".join(
        f"[{row.get('chunk_id', row.get('evidence_id', 'chunk'))}] {row.get('text', '')}"
        for row in evidence
    )
    sections = ["Decision state: active", hypothesis_text]
    if rendered_evidence:
        sections.extend(["Evidence supporting this decision:", rendered_evidence])
    return "\n".join(sections).strip()


def retrieval_generation_context(item: dict) -> str:
    """Evidence-only context for controls that do not emit a revision decision."""

    return "\n".join(
        f"[{row.get('chunk_id', row.get('evidence_id', 'chunk'))}] {row.get('text', '')}"
        for row in item.get("evidence", [])
        if isinstance(row, dict)
    ).strip()


def render_answer(item: dict, result: dict, baseline: str, local_generator) -> tuple[str, int | None, int | None, int]:
    """Render one answer under the decision-render contract.

    Non-active structured states are deterministic and never call the language
    model. Stateless controls still call the shared generator from raw evidence.
    """

    if baseline in UNSTRUCTURED_RENDER_BASELINES:
        generated = local_generator.generate(
            item["query"],
            retrieval_generation_context(item),
            instruction=RETRIEVAL_RENDER_INSTRUCTION,
        )
        return generated.text, generated.input_tokens, generated.output_tokens, 1

    state = str(result.get("predicted_state") or "").lower()
    if state != "active" or not result.get("predicted_hypothesis_id"):
        answer = NON_ACTIVE_ANSWERS.get(state, "UNKNOWN: no structured answer was selected.")
        return answer, 0, 0, 0

    generated = local_generator.generate(
        item["query"],
        generation_context(item, result),
        instruction=DECISION_RENDER_INSTRUCTION,
    )
    return generated.text, generated.input_tokens, generated.output_tokens, 1


def structured_answer_correct(item: dict, result: dict) -> bool | None:
    if result.get("answer_correct") is not None:
        return bool(result["answer_correct"])
    gold_state = item.get("gold_state")
    predicted_state = result.get("predicted_state")
    if gold_state is None or predicted_state is None:
        return None
    if str(gold_state).lower() != str(predicted_state).lower():
        return False
    gold_hypothesis = item.get("gold_hypothesis_id")
    if gold_hypothesis is not None:
        return gold_hypothesis == result.get("predicted_hypothesis_id")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator", default="unspecified")
    parser.add_argument("--local-model", help="Open-weight model served by Ollama, vLLM, or llama.cpp")
    parser.add_argument("--local-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--model-timeout", type=float, default=120.0)
    parser.add_argument("--no-generation", action="store_true", help="Retain structured-only legacy execution")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    base_config = json.loads(args.config.read_text()) if args.config else {}
    local_generator = None if args.no_generation else build_local_generator(args)
    baseline = build_baseline(args.baseline, dict(base_config))
    run_id = uuid.uuid4().hex
    rows = [json.loads(line) for line in args.fixture.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in rows:
            start = time.perf_counter()
            if item.get("hypotheses") is not None and hasattr(baseline, "config"):
                baseline.config["hypotheses"] = item["hypotheses"]
            with measure_revision() as revision_cost:
                result = baseline.process_turn(
                    item.get("session_id", item["base_example_id"]),
                    item["query"],
                    evidence_for_baseline(item),
                    item.get("target_time"),
                )
            if local_generator is not None and args.baseline in LOCAL_RENDER_BASELINES:
                answer, input_tokens, output_tokens, model_calls = render_answer(
                    item, result, args.baseline, local_generator
                )
                result["answer"] = answer
                result["input_tokens"] = input_tokens
                result["output_tokens"] = output_tokens
                result["model_calls"] = int(result.get("model_calls", 0)) + model_calls
            metadata = {
                "manifest": baseline.manifest(),
                "verification": result.get("verification"),
                "revision": result.get("revision"),
                "native_execution": result.get("native_execution"),
                "transformation_metadata": item.get("transformation_metadata"),
                "generator_manifest": local_generator.manifest() if local_generator else None,
            }
            metadata["revision_latency_ms"] = revision_cost["revision_latency_ms"]
            metadata["peak_memory_mb"] = revision_cost["peak_memory_mb"]
            measured_nli_calls = resolve_nli_calls(baseline, result)
            if measured_nli_calls is not None:
                metadata["nli_calls"] = measured_nli_calls
            if local_generator is not None:
                # Local open-weight inference under the plan's $0.00 budget: a
                # measured zero, not an absent measurement.
                metadata["estimated_cost_usd"] = 0.0
                metadata["generation_contract"] = GENERATION_CONTRACT_VERSION
            transformation_metadata = item.get("transformation_metadata") or {}
            for key in (
                "gold_answer",
                "gold_answer_aliases",
                "gold_claims",
                "claim_gold_labels",
                "citation_gold_labels",
                "temporal_slice",
                "stale_hypothesis_id",
                "annotation_status",
                "human_review_status",
            ):
                if transformation_metadata.get(key) is not None:
                    metadata[key] = transformation_metadata[key]
            for key in (
                "condition",
                "ablation",
                "mechanism",
                "threshold",
                "support_threshold",
                "decision_threshold",
                "estimated_cost_usd",
                "model_calls",
                "nli_calls",
                "peak_memory_mb",
            ):
                if base_config.get(key) is not None:
                    metadata[key] = base_config[key]
                if result.get(key) is not None:
                    metadata[key] = result[key]
            record = PredictionRecord(
                run_id=run_id, example_id=item["example_id"], base_example_id=item["base_example_id"],
                pair_id=item.get("pair_id"), dataset=item.get("dataset", "river_bench"),
                source_dataset=item.get("source_dataset", "unknown"), split=item.get("split", "test"),
                intervention=item.get("intervention", "none"), baseline=args.baseline,
                comparison_mode=result.get("comparison_mode", base_config.get("comparison_mode", "controlled")),
                baseline_version=baseline.version, generator=args.generator, seed=args.seed,
                gold_state=item.get("gold_state"), gold_hypothesis_id=item.get("gold_hypothesis_id"),
                predicted_state=result.get("predicted_state"), predicted_hypothesis_id=result.get("predicted_hypothesis_id"),
                answer=result.get("answer"),
                intervention_hypothesis_id=item.get("intervention_hypothesis_id"),
                answer_correct=structured_answer_correct(item, result), citation_supported=result.get("citation_supported"),
                confidence=result.get("confidence"),
                state_probabilities=result.get("state_probabilities"),
                claim_verification_results=result.get("claim_verification_results"),
                claim_verification_result_refs=result.get("claim_verification_result_refs"),
                latency_ms=(time.perf_counter() - start) * 1000,
                input_tokens=result.get("input_tokens"), output_tokens=result.get("output_tokens"),
                configuration_hash=digest(base_config), code_commit=git_commit(), dataset_hash=digest(rows),
                prompt_hash=digest({
                    "prompt": base_config.get("prompt", ""),
                    "generator": local_generator.manifest() if local_generator else None,
                    "generation_contract": GENERATION_CONTRACT_VERSION if local_generator else None,
                    "decision_instruction": DECISION_RENDER_INSTRUCTION if local_generator else None,
                    "retrieval_instruction": RETRIEVAL_RENDER_INSTRUCTION if local_generator else None,
                }),
                metadata=metadata,
            )
            handle.write(record.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
