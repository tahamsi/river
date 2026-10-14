"""Render RIVER-Bench into browsable Markdown review pages.

The split files are JSONL and too large for a code-hosting web view, so human
review has to happen somewhere else. This writes one Markdown page per batch of
pairs, grouped by intervention family, with both variants of a pair adjacent:
the paired design is what a reviewer checks, so splitting a pair across pages
would hide the thing under review.

Each example shows the query, the target time, the gold state, and the evidence
in arrival order with both time axes visible, because the central claim of the
benchmark is that arrival order must not drive the answer. The source-native
record is included truncated; the untruncated record stays in the JSONL.

Usage:
    python scripts/benchmark/build_review_pack.py [--grounded-dir data/grounded]
                                        [--out data/grounded/review]
                                        [--pairs-per-page 25]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SPLITS = ("train", "dev", "test")
# Enough of the source record to check a claim against, without making the page
# too large for a web view to render.
SOURCE_RECORD_CHARS = 1200


def load_split(grounded_dir: Path, split: str) -> list[dict[str, Any]]:
    path = grounded_dir / f"river_bench_{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing split file: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def escape_cell(text: str) -> str:
    """Make a string safe inside a Markdown table cell."""

    return text.replace("|", "\\|").replace("\n", " ").strip()


def interval(evidence: dict[str, Any]) -> str:
    valid = evidence.get("valid_time") or {}
    start = valid.get("start") or "—"
    end = valid.get("end") or "open"
    return f"{start} → {end}"


def render_evidence(example: dict[str, Any]) -> list[str]:
    rows = example.get("evidence") or []
    if not rows:
        return ["_No evidence in this bundle._", ""]
    # Arrival order is the reviewer's frame of reference: the interventions are
    # defined by what arrives when.
    ordered = sorted(rows, key=lambda item: str(item.get("transaction_time") or ""))
    lines = [
        "| # | Arrived (transaction time) | Valid interval | Hypothesis | Text |",
        "|---|---|---|---|---|",
    ]
    for position, item in enumerate(ordered, start=1):
        lines.append(
            "| {n} | `{arrived}` | `{valid}` | `{hyp}` | {text} |".format(
                n=position,
                arrived=item.get("transaction_time") or "—",
                valid=interval(item),
                hyp=item.get("hypothesis_id") or "—",
                text=escape_cell(str(item.get("text", ""))),
            )
        )
    lines.append("")
    return lines


def render_hypotheses(example: dict[str, Any]) -> list[str]:
    hypotheses = example.get("hypotheses") or []
    if not hypotheses:
        return ["_No hypotheses._", ""]
    gold = example.get("gold_hypothesis_id")
    lines = ["| Hypothesis | Object | Gold? | Supported by |", "|---|---|---|---|"]
    for hypothesis in hypotheses:
        identifier = hypothesis.get("hypothesis_id")
        lines.append(
            "| `{hid}` | {obj} | {gold} | {ev} |".format(
                hid=identifier,
                obj=escape_cell(str(hypothesis.get("object", ""))),
                gold="**yes**" if identifier == gold else "",
                ev=", ".join(f"`{eid}`" for eid in hypothesis.get("evidence_ids") or []) or "—",
            )
        )
    lines.append("")
    return lines


def render_example(example: dict[str, Any]) -> list[str]:
    metadata = example.get("transformation_metadata") or {}
    lines = [
        f"#### `{example['example_id']}` — variant role `{metadata.get('variant_role', '—')}`",
        "",
        f"- **Query:** {example.get('query', '')}",
        f"- **Target time:** `{example.get('target_time') or '—'}`",
        f"- **Gold state:** `{example.get('gold_state')}`",
        f"- **Gold hypothesis:** `{example.get('gold_hypothesis_id') or 'none (no single hypothesis is correct)'}`",
        f"- **Split:** `{example.get('split')}` · **Source:** `{example.get('source_dataset')}`",
        "",
        "**Evidence, in arrival order**",
        "",
    ]
    lines += render_evidence(example)
    lines += ["**Hypotheses**", ""]
    lines += render_hypotheses(example)

    native = metadata.get("source_native_record")
    if native is not None:
        rendered = json.dumps(native, indent=1, ensure_ascii=False)
        truncated = len(rendered) > SOURCE_RECORD_CHARS
        if truncated:
            rendered = rendered[:SOURCE_RECORD_CHARS] + "\n… truncated; full record is in the JSONL"
        lines += [
            "<details><summary>Source-native record (check every claim against this)</summary>",
            "",
            "```json",
            rendered,
            "```",
            "",
            "</details>",
            "",
        ]
    return lines


def render_pair(pair_id: str, examples: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(examples, key=lambda item: item["example_id"])
    lines = [
        f"### Pair `{pair_id}`",
        "",
        "- [ ] Reviewed and approved — both variants are source-grounded",
        "",
    ]
    for example in ordered:
        lines += render_example(example)
    lines += ["---", ""]
    return lines


def write_pages(
    families: dict[str, dict[str, list[dict[str, Any]]]],
    out_dir: Path,
    pairs_per_page: int,
) -> list[tuple[str, str, int]]:
    """Write one or more pages per family. Returns index rows."""

    index_rows: list[tuple[str, str, int]] = []
    for family in sorted(families):
        pairs = sorted(families[family].items())
        pages = [pairs[start : start + pairs_per_page] for start in range(0, len(pairs), pairs_per_page)]
        for number, page in enumerate(pages, start=1):
            name = f"{family}-{number:02d}.md"
            lines = [
                f"# {family} — page {number} of {len(pages)}",
                "",
                "Check each pair against its source-native record: the answer, every",
                "claim label, the citation, the temporal interval, the gold state and the",
                "selected hypothesis. Reject the whole pair if any required fact is not",
                "source-grounded. Tick the box on a pair only when both variants pass.",
                "",
                "---",
                "",
            ]
            for pair_id, examples in page:
                lines += render_pair(pair_id, examples)
            (out_dir / name).write_text("\n".join(lines), encoding="utf-8")
            index_rows.append((family, name, len(page)))
    return index_rows


def write_index(index_rows: list[tuple[str, str, int]], out_dir: Path, total_pairs: int) -> None:
    lines = [
        "# RIVER-Bench human review pack",
        "",
        f"{total_pairs} pairs (1,500 examples) awaiting review. Every row currently carries",
        "`annotation_status: rebuilt_pending_human_review` and `human_review_status: pending`;",
        "no approval is recorded anywhere until a person grants it.",
        "",
        "Pages are grouped by intervention family and split so each one renders in a",
        "browser. Tick a pair's checkbox by editing the page and committing the change.",
        "",
        "| Family | Page | Pairs |",
        "|---|---|---|",
    ]
    for family, name, count in index_rows:
        lines.append(f"| `{family}` | [{name}]({name}) | {count} |")
    lines += ["", f"**Total pages:** {len(index_rows)}", ""]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounded-dir", type=Path, default=Path("data/grounded"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pairs-per-page", type=int, default=25)
    args = parser.parse_args()

    out_dir = args.out or (args.grounded_dir / "review")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.md"):
        stale.unlink()

    families: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    total = 0
    for split in SPLITS:
        for example in load_split(args.grounded_dir, split):
            families[example["intervention"]][example["pair_id"]].append(example)
            total += 1

    index_rows = write_pages(families, out_dir, args.pairs_per_page)
    total_pairs = sum(len(pairs) for pairs in families.values())
    write_index(index_rows, out_dir, total_pairs)
    print(f"{total} examples, {total_pairs} pairs, {len(index_rows)} pages -> {out_dir}")


if __name__ == "__main__":
    main()
