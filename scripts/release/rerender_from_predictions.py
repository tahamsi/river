"""Re-render answers from frozen structured predictions without retraining.

The input prediction file remains immutable. Structured state, hypothesis,
confidence and model outputs are copied exactly; only generation-dependent
fields and run provenance are replaced under the decision-render generation contract.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from experiments.run import (
    DECISION_RENDER_INSTRUCTION,
    GENERATION_CONTRACT_VERSION,
    RETRIEVAL_RENDER_INSTRUCTION,
    digest,
    git_commit,
    render_answer,
)
from river.generation import LocalOpenAICompatibleGenerator


SCORING_FIELDS = {
    "answer_text_correct",
    "answer_named_hypothesis_id",
    "answer_contract_valid",
    "direct_answer",
    "answer_decision_faithful",
    "end_to_end_answer_correct",
}


def load_fixture(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, {row["example_id"]: row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generator-name", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--expected-examples", type=int)
    args = parser.parse_args()

    source_rows = [
        json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    fixture_rows, fixture = load_fixture(args.fixture)
    if args.expected_examples is not None and len(source_rows) != args.expected_examples:
        raise SystemExit(f"expected {args.expected_examples} source rows, found {len(source_rows)}")
    if len({row["example_id"] for row in source_rows}) != len(source_rows):
        raise SystemExit("source predictions contain duplicate example IDs")
    missing = sorted({row["example_id"] for row in source_rows} - fixture.keys())
    if missing:
        raise SystemExit(f"fixture is missing {len(missing)} prediction IDs; first: {missing[:3]}")

    seed = int(source_rows[0].get("seed", 42)) if source_rows else 42
    generator = LocalOpenAICompatibleGenerator(
        args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        timeout=args.timeout,
        seed=seed,
    )
    run_id = uuid.uuid4().hex
    commit = git_commit()
    if "uncommitted" in commit or "+dirty:" in commit:
        raise SystemExit(f"release rerender requires a clean commit; got {commit}")
    prompt_hash = digest(
        {
            "generator": generator.manifest(),
            "generation_contract": GENERATION_CONTRACT_VERSION,
            "decision_instruction": DECISION_RENDER_INSTRUCTION,
            "retrieval_instruction": RETRIEVAL_RENDER_INSTRUCTION,
        }
    )

    output_rows: list[dict] = []
    for row in source_rows:
        item = fixture[row["example_id"]]
        result = {
            "predicted_state": row.get("predicted_state"),
            "predicted_hypothesis_id": row.get("predicted_hypothesis_id"),
            "revision": (row.get("metadata") or {}).get("revision"),
            "model_calls": 0,
        }
        started = time.perf_counter()
        answer, input_tokens, output_tokens, model_calls = render_answer(
            item, result, str(row.get("baseline")), generator
        )
        updated = dict(row)
        for field in SCORING_FIELDS:
            updated.pop(field, None)
        updated.update(
            run_id=run_id,
            generator=args.generator_name,
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=float((row.get("metadata") or {}).get("revision_latency_ms", 0.0))
            + (time.perf_counter() - started) * 1000,
            prompt_hash=prompt_hash,
            code_commit=commit,
        )
        metadata = dict(updated.get("metadata") or {})
        metadata.update(
            generator_manifest=generator.manifest(),
            generation_contract=GENERATION_CONTRACT_VERSION,
            rerendered_from_run_id=row.get("run_id"),
            model_calls=model_calls,
            estimated_cost_usd=0.0,
        )
        updated["metadata"] = metadata
        output_rows.append(updated)

    if [row["example_id"] for row in output_rows] != [row["example_id"] for row in source_rows]:
        raise SystemExit("row order or IDs changed during rerender")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {len(output_rows)} rows to {args.output}")
    print(f"fixture digest: {digest(fixture_rows)}")
    print(f"generation contract: {GENERATION_CONTRACT_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
