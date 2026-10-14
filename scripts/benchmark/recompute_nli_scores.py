"""Measure the entailment, neutral and contradiction scores with one NLI model.

The premise is the evidence chunk and the hypothesis is the claim under test,
phrased the same way for every row. All three fields therefore come from a
single measurement on a single scale. A support-only verifier such as MiniCheck
cannot provide a contradiction score.

Output is written to a new directory; existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SPLITS = ("train", "dev", "test")
# A plain MNLI model. Deliberately not one fine-tuned on FEVER: FEVER is one of
# this benchmark's source corpora, and a verifier trained on it would carry an
# advantage on those rows that has nothing to do with the evidence.
DEFAULT_MODEL = "microsoft/deberta-large-mnli"


def claim_text(row: dict, hypothesis: dict) -> str:
    """A declarative hypothesis for the NLI pair.

    The stored subject is a source identifier rather than language, so the query
    supplies the relation. One phrasing for every row keeps the scores
    comparable across families.
    """

    query = str(row.get("query") or "").strip().rstrip("?").strip()
    obj = str(hypothesis.get("object") or "").strip()
    if not query:
        return obj
    return f"The answer to the question '{query}?' is {obj}."


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def leak_rate(rows: list[dict]) -> tuple[float, int]:
    hits = []
    for row in rows:
        gold = row.get("gold_hypothesis_id")
        if gold is None:
            continue
        best: dict[str, float] = {}
        for evidence in row.get("evidence") or []:
            key = evidence.get("hypothesis_id")
            if key is None:
                continue
            best[key] = max(best.get(key, -1.0), float(evidence.get("entailment") or 0.0))
        if len(best) >= 2:
            hits.append(max(best, key=lambda k: best[k]) == gold)
    return (sum(hits) / len(hits) if hits else float("nan")), len(hits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounded-dir", type=Path, default=Path("data/grounded"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/grounded_nli"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    # Label order varies between MNLI checkpoints, so read it rather than assume.
    order = {name.lower(): index for index, name in model.config.id2label.items()}
    missing = {"entailment", "neutral", "contradiction"} - set(order)
    if missing:
        raise SystemExit(f"{args.model} does not expose {missing}; id2label={model.config.id2label}")
    print(f"{args.model} on {device}, label order {model.config.id2label}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_counts = {}

    for split in SPLITS:
        rows = [
            json.loads(line)
            for line in (args.grounded_dir / f"river_bench_{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        before, contested = leak_rate(rows)

        premises: list[str] = []
        hypotheses: list[str] = []
        targets: list[dict] = []
        for row in rows:
            by_id = {item["hypothesis_id"]: item for item in row.get("hypotheses") or []}
            for evidence in row.get("evidence") or []:
                hypothesis = by_id.get(evidence.get("hypothesis_id"))
                if hypothesis is None or not evidence.get("text"):
                    continue
                premises.append(str(evidence["text"]))
                hypotheses.append(claim_text(row, hypothesis))
                targets.append(evidence)

        print(f"{split}: {len(rows)} rows, {len(targets)} pairs", flush=True)
        for start in range(0, len(targets), args.batch_size):
            batch_premises = premises[start:start + args.batch_size]
            batch_hypotheses = hypotheses[start:start + args.batch_size]
            encoded = tokenizer(
                batch_premises, batch_hypotheses,
                return_tensors="pt", truncation=True, padding=True, max_length=512,
            ).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(**encoded).logits, dim=-1).cpu()
            for offset, distribution in enumerate(probabilities):
                evidence = targets[start + offset]
                evidence["entailment"] = round(float(distribution[order["entailment"]]), 6)
                evidence["neutral"] = round(float(distribution[order["neutral"]]), 6)
                evidence["contradiction"] = round(float(distribution[order["contradiction"]]), 6)
                evidence["verifier_scores_measured_by"] = args.model
                evidence.pop("contradiction_is_assigned", None)

        after, _ = leak_rate(rows)
        destination = args.output_dir / f"river_bench_{split}.jsonl"
        destination.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest_counts[split] = len(rows)

        distinct = {
            field: len({round(float(e.get(field) or 0.0), 4) for r in rows for e in r.get("evidence") or []})
            for field in ("entailment", "neutral", "contradiction")
        }
        print(
            f"{split}: argmax-entailment picks gold {before:.3f} -> {after:.3f} over {contested} contested; "
            f"distinct {distinct}",
            flush=True,
        )

    combined = []
    for split in SPLITS:
        combined.extend(
            json.loads(line)
            for line in (args.output_dir / f"river_bench_{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    (args.output_dir / "river_bench_1500.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in combined), encoding="utf-8"
    )

    manifest = {
        "schema_version": "river-bench-grounded-v3-nli",
        "derived_from": str(args.grounded_dir),
        "example_count": len(combined),
        "split_example_counts": manifest_counts,
        "verifier_scores": {
            "source": "measured",
            "model": args.model,
            "task": "natural language inference, three-way",
            "premise": "the evidence chunk text",
            "hypothesis": "The answer to the question '<query>?' is <object>.",
            "note": (
                "entailment, neutral and contradiction all come from this model on the "
                "same pair, so all three are measured on one scale. No field is "
                "assigned."
            ),
            "regenerate_with": "scripts/benchmark/recompute_nli_scores.py",
        },
        "output_sha256": {
            "combined": digest(args.output_dir / "river_bench_1500.jsonl"),
            **{s: digest(args.output_dir / f"river_bench_{s}.jsonl") for s in SPLITS},
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
