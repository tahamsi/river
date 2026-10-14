"""How much of the benchmark a model can solve without reading any verifier score.

scripts/diagnostics/leak_baseline.py asks what the stored entailment values give away. This
asks the complementary question: strip the three NLI fields and
train a classifier on everything else RIVER-L sees (valid-time applicability,
interval shape, provenance, extraction confidence, independence grouping,
evidence completeness) plus the bookkeeping shape of the example (how many
hypotheses, how much evidence each carries). Whatever it scores is template
signal: structure the construction leaves behind that says the answer without
any claim being checked against any evidence.

The same classifier is also trained with the NLI fields put back, so the gap
between the two rows is what the measured verifier scores contribute.

Arrival order is never a feature. It is the thing the benchmark is built to make
uninformative, and RIVER-L is forbidden from reading it too.

Training is on train, the regularisation strength is chosen on dev, and test is
read once per feature set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from river.learned.features import RiverLFeatureExtractor, target_time_from_example

NLI_FEATURES = {"entailment_probability", "contradiction_probability", "neutral_probability"}
STATES = ["active", "disputed", "evolution", "abstain", "unknown"]
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def per_evidence(row: dict, extractor: RiverLFeatureExtractor, keep: list[int]) -> list[tuple[str | None, list[float]]]:
    evidence = [item for item in row.get("evidence") or [] if isinstance(item, dict)]
    groups = Counter(str(item.get("independence_group", item.get("evidence_id", ""))) for item in evidence)
    target = target_time_from_example(row)
    out = []
    for item in evidence:
        vector = extractor.evidence_features(item, group_counts=groups, target_time=target)
        out.append((item.get("hypothesis_id"), [vector[i] for i in keep]))
    return out


def pool(vectors: list[list[float]], width: int) -> list[float]:
    if not vectors:
        return [0.0] * (3 * width)
    columns = list(zip(*vectors))
    return [sum(c) / len(c) for c in columns] + [max(c) for c in columns] + [min(c) for c in columns]


def row_vector(row: dict, extractor, keep) -> list[float]:
    items = per_evidence(row, extractor, keep)
    width = len(keep)
    by_hypothesis: dict[str, list[list[float]]] = defaultdict(list)
    for hypothesis, vector in items:
        if hypothesis is not None:
            by_hypothesis[hypothesis].append(vector)
    counts = sorted((len(v) for v in by_hypothesis.values()), reverse=True)
    shape = [
        float(len(row.get("hypotheses") or [])),
        float(len(items)),
        float(len(by_hypothesis)),
        float(counts[0]) if counts else 0.0,
        float(counts[1]) if len(counts) > 1 else 0.0,
        float(sum(1 for hypothesis, _ in items if hypothesis is None)),
    ]
    return pool([v for _, v in items], width) + shape


def hypothesis_vectors(row: dict, extractor, keep) -> tuple[list[str], list[list[float]]]:
    items = per_evidence(row, extractor, keep)
    width = len(keep)
    ids = [h.get("hypothesis_id") if isinstance(h, dict) else h for h in row.get("hypotheses") or []]
    ids = [i for i in ids if i is not None] or sorted({h for h, _ in items if h is not None})
    # The benchmark lists the gold hypothesis in a fixed position per family, so
    # any order-sensitive tie-break would read the label off the list. Order by
    # a hash of the id instead.
    ids = sorted(ids, key=lambda i: hashlib.sha1(f"{row.get('example_id')}|{i}".encode()).hexdigest())
    total = max(len(items), 1)
    vectors = []
    for hypothesis in ids:
        own = [v for h, v in items if h == hypothesis]
        vectors.append(pool(own, width) + [float(len(own)), len(own) / total])
    return ids, vectors


def standardise(train: torch.Tensor, *others: torch.Tensor):
    mean, std = train.mean(0), train.std(0).clamp_min(1e-6)
    return [(t - mean) / std for t in (train, *others)]


def fit_state(x, y, classes, weight_decay, hidden):
    torch.manual_seed(13)
    if hidden:
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, classes))
    else:
        model = torch.nn.Linear(x.shape[1], classes)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    for _ in range(600):
        optimiser.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimiser.step()
    return model


def pad(groups):
    """Stack variable-length hypothesis sets so one forward pass scores them all."""

    width = max(x.shape[0] for x, _ in groups)
    x = torch.zeros(len(groups), width, groups[0][0].shape[1])
    mask = torch.zeros(len(groups), width, dtype=torch.bool)
    for i, (vectors, _) in enumerate(groups):
        x[i, : vectors.shape[0]] = vectors
        mask[i, : vectors.shape[0]] = True
    return x, mask, torch.tensor([g for _, g in groups])


def scores(model, x, mask):
    return model(x).squeeze(-1).masked_fill(~mask, float("-inf"))


def fit_selector(groups, weight_decay):
    torch.manual_seed(13)
    x, mask, gold = pad(groups)
    model = torch.nn.Linear(x.shape[2], 1)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    for _ in range(300):
        optimiser.zero_grad()
        torch.nn.functional.cross_entropy(scores(model, x, mask), gold).backward()
        optimiser.step()
    return model


def run(splits: dict[str, list[dict]], use_nli: bool) -> dict:
    extractor = RiverLFeatureExtractor()
    keep = [i for i, name in enumerate(extractor.feature_names) if use_nli or name not in NLI_FEATURES]
    names = [extractor.feature_names[i] for i in keep]

    tensors = {s: torch.tensor([row_vector(r, extractor, keep) for r in rows], dtype=torch.float32) for s, rows in splits.items()}
    labels = {s: torch.tensor([STATES.index(str(r["gold_state"]).lower()) for r in rows]) for s, rows in splits.items()}
    xtr, xdv, xte = standardise(tensors["train"], tensors["dev"], tensors["test"])

    state = {}
    for label, hidden in (("logistic", 0), ("mlp", 64)):
        best = None
        for weight_decay in WEIGHT_DECAYS:
            model = fit_state(xtr, labels["train"], len(STATES), weight_decay, hidden)
            with torch.no_grad():
                dev = (model(xdv).argmax(1) == labels["dev"]).float().mean().item()
            if best is None or dev > best[0]:
                best = (dev, weight_decay, model)
        with torch.no_grad():
            predicted = best[2](xte).argmax(1)
        correct = int((predicted == labels["test"]).sum())
        state[label] = {"dev": round(best[0], 4), "weight_decay": best[1], "test": round(correct / len(predicted), 4), "correct": correct, "n": len(predicted)}

    selection = {}
    prepared = {}
    for split, rows in splits.items():
        groups = []
        for row in rows:
            gold = row.get("gold_hypothesis_id")
            ids, vectors = hypothesis_vectors(row, extractor, keep)
            if gold is None or gold not in ids or len(ids) < 2:
                continue
            groups.append((torch.tensor(vectors, dtype=torch.float32), ids.index(gold)))
        prepared[split] = groups
    stacked = torch.cat([x for x, _ in prepared["train"]])
    mean, std = stacked.mean(0), stacked.std(0).clamp_min(1e-6)
    norm = {s: [((x - mean) / std, g) for x, g in groups] for s, groups in prepared.items()}

    def accuracy(model, groups):
        # Exact ties earn 1/k rather than whichever tied hypothesis comes first.
        x, mask, gold = pad(groups)
        with torch.no_grad():
            s = scores(model, x, mask)
            top = s == s.max(1, keepdim=True).values
            hits = float((top[torch.arange(len(gold)), gold].float() / top.sum(1)).sum())
        return hits, len(groups)

    best = None
    for weight_decay in WEIGHT_DECAYS:
        model = fit_selector(norm["train"], weight_decay)
        hits, n = accuracy(model, norm["dev"])
        if best is None or hits / n > best[0]:
            best = (hits / n, weight_decay, model)
    hits, n = accuracy(best[2], norm["test"])
    chance = sum(1 / x.shape[0] for x, _ in norm["test"]) / n
    selection = {"dev": round(best[0], 4), "weight_decay": best[1], "test": round(hits / n, 4), "correct": round(hits, 1), "n": n, "uniform_chance": round(chance, 4)}
    return {"features": names, "state_accuracy": state, "hypothesis_selection": selection}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/grounded_human_frozen"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Runs beside GPU inference; a handful of CPU threads is plenty for this size.
    torch.set_num_threads(4)

    splits ={s: load(args.data / f"river_bench_{s}.jsonl") for s in ("train", "dev", "test")}
    test_states = Counter(str(r["gold_state"]).lower() for r in splits["test"])
    majority_state, majority_count = test_states.most_common(1)[0]
    train_majority = Counter(str(r["gold_state"]).lower() for r in splits["train"]).most_common(1)[0][0]

    report = {
        "data": str(args.data),
        "rows": {s: len(r) for s, r in splits.items()},
        "test_state_distribution": dict(test_states),
        "majority_class_from_train": {"state": train_majority, "test": round(test_states[train_majority] / len(splits["test"]), 4)},
        "without_nli_fields": run(splits, use_nli=False),
        "with_nli_fields": run(splits, use_nli=True),
    }

    print(f"{args.data}: train {len(splits['train'])}, dev {len(splits['dev'])}, test {len(splits['test'])}")
    print(f"test gold states: {dict(test_states)}")
    print(f"majority class (from train, '{train_majority}'): state accuracy {report['majority_class_from_train']['test']:.4f}\n")
    print(f"{'feature set':22s} {'model':9s} {'state acc':>9s}  {'selection':>9s}  (dev-selected weight decay)")
    for key, label in (("without_nli_fields", "no NLI fields"), ("with_nli_fields", "with NLI fields")):
        block = report[key]
        for model, result in block["state_accuracy"].items():
            sel = block["hypothesis_selection"]
            print(f"{label:22s} {model:9s} {result['test']:9.4f}  {sel['test']:9.4f}  (state wd {result['weight_decay']}, select wd {sel['weight_decay']})")
    sel = report["without_nli_fields"]["hypothesis_selection"]
    print(f"\nselection is over {sel['n']} test rows with a gold hypothesis among two or more; uniform chance {sel['uniform_chance']:.4f}")
    print("features without NLI:", ", ".join(report["without_nli_fields"]["features"]), "+ per-example shape counts")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
