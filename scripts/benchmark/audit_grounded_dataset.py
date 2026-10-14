"""Mechanically audit the grounded benchmark against its source records.

Human review has to judge whether a constructed example is a fair test. This
checks the part that needs no judgement: that every string the benchmark
presents can be found in the record it claims to come from, that a pair's two
variants really are the same facts in a different order, and that the gold
answer is not recoverable from anything but the evidence.

Findings are reported per check with example ids, so a reviewer reads source
records only where a machine cannot settle the question. Nothing here records
an approval; the review status in the data is untouched.

Usage:
    python scripts/benchmark/audit_grounded_dataset.py [--grounded-dir data/grounded]
                                             [--report data/grounded/audit_findings.md]
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SPLITS = ("train", "dev", "test")

# Families whose gold state is fixed by the construction, so a row carrying a
# different state is a build fault rather than a judgement call.
EXPECTED_STATE = {
    "concurrent_disagreement": {"disputed"},
    "insufficient_evidence_nei": {"abstain", "unknown"},
    "unsupported_late_assertion": {"active", "abstain"},
    "order_reversal": {"active"},
    "expired_evidence": {"active"},
    "temporal_evolution": {"evolution", "active"},
}


def normalise(text: str) -> str:
    """Fold whitespace, quotes and case so string containment is meaningful."""

    folded = unicodedata.normalize("NFKC", str(text or ""))
    folded = folded.replace("’", "'").replace("“", '"').replace("”", '"')
    # The source corpora contain tokenizer artefacts ("Fly- Away Baby"), so
    # spacing cannot be treated as significant.
    return re.sub(r"[\s]+", " ", folded).strip().casefold()


def source_strings(record: Any) -> list[str]:
    """Every string anywhere in a source record, for containment checks."""

    found: list[str] = []
    stack = [record]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.append(normalise(item))
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return found


def contained(needle: str, haystacks: Iterable[str]) -> bool:
    target = normalise(needle)
    if not target:
        return False
    stripped = re.sub(r"[^\w ]+", "", target)
    for candidate in haystacks:
        if target in candidate:
            return True
        # Punctuation is inconsistently spaced in the source corpora, so a
        # punctuation-insensitive fallback avoids false alarms.
        if stripped and stripped in re.sub(r"[^\w ]+", "", candidate):
            return True
    return False


def grounding_tier(text: str, obj: str, haystack: list[str]) -> str:
    """Grade one evidence item against the record it cites.

    ``verbatim``   the sentence itself appears in the source;
    ``templated``  the sentence is generated, but the fact it asserts is in the
                   source, so a reader can still check it;
    ``authored``   neither, so nothing in the source record supports the text.
    """

    if contained(text, haystack):
        return "verbatim"
    if obj and contained(obj, haystack):
        return "templated"
    return "authored"


class Findings:
    """Collects issues keyed by check name, capped for a readable report."""

    LIMIT = 8

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = defaultdict(list)
        self.checked: Counter[str] = Counter()
        self.tiers: dict[str, Counter[str]] = defaultdict(Counter)

    def tier(self, family: str, name: str) -> None:
        self.tiers[family][name] += 1

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.checked[name] += 1
        if ok:
            return
        self.counts[name] += 1
        if len(self.examples[name]) < self.LIMIT:
            self.examples[name].append(detail)


def audit_row(row: dict[str, Any], findings: Findings) -> None:
    metadata = row.get("transformation_metadata") or {}
    native = metadata.get("source_native_record")
    identifier = row["example_id"]
    family = row["intervention"]

    if not isinstance(native, (dict, list)):
        findings.check("source record present", False, f"{identifier}: no source_native_record")
        return
    findings.check("source record present", True, "")
    haystack = source_strings(native)

    # Evidence prose is not always quoted: some families state a source fact in a
    # template sentence. That is defensible, but it has to be visible, so each
    # item is graded rather than passed or failed.
    for item in row.get("evidence") or []:
        text = str(item.get("text", ""))
        tier = grounding_tier(text, str(item.get("object", "")), haystack)
        findings.tier(row["intervention"], tier)
        findings.check(
            "evidence is anchored in the source record",
            tier != "authored",
            f"{identifier} [{family}]: {text[:90]!r}",
        )

    # The gold answer must likewise be stated by the source.
    gold_id = row.get("gold_hypothesis_id")
    objects = {h["hypothesis_id"]: str(h.get("object", "")) for h in row.get("hypotheses") or []}
    if gold_id and gold_id in objects:
        findings.check(
            "gold answer is in the source record",
            contained(objects[gold_id], haystack),
            f"{identifier} [{family}]: gold={objects[gold_id][:70]!r}",
        )

    # An alternative that renames the gold answer is not an alternative.
    if gold_id and gold_id in objects and len(objects) > 1:
        gold_text = normalise(objects[gold_id])
        for other_id, other in objects.items():
            if other_id == gold_id:
                continue
            other_text = normalise(other)
            # "non-selective" contains "selective" but states the opposite, so
            # containment alone would reject a genuine disagreement.
            negated = any(
                other_text == prefix + gold_text or gold_text == prefix + other_text
                for prefix in ("non-", "non", "un", "in", "im", "not ", "no ")
            )
            distinct = negated or not (
                other_text == gold_text or other_text in gold_text or gold_text in other_text
            )
            findings.check(
                "alternative differs from the gold answer",
                distinct,
                f"{identifier}: gold={objects[gold_id][:40]!r} alt={other[:40]!r}",
            )

    # Evidence must be attributable to a hypothesis the row actually offers.
    known = set(objects)
    for item in row.get("evidence") or []:
        hypothesis = item.get("hypothesis_id")
        findings.check(
            "evidence points at a declared hypothesis",
            hypothesis is None or hypothesis in known,
            f"{identifier}: evidence cites {hypothesis!r}",
        )

    # Arrival stamps must be readable and ordered, since interventions are
    # defined by what arrives when.
    stamps = [item.get("transaction_time") for item in row.get("evidence") or []]
    findings.check(
        "every evidence item has an arrival time",
        all(stamps),
        f"{identifier}: {sum(1 for s in stamps if not s)} items without transaction_time",
    )

    expected = EXPECTED_STATE.get(family)
    if expected:
        findings.check(
            "gold state matches the intervention family",
            row.get("gold_state") in expected,
            f"{identifier}: {family} has gold_state={row.get('gold_state')!r}",
        )

    findings.check(
        "query is non-empty",
        bool(str(row.get("query", "")).strip()),
        f"{identifier}: empty query",
    )


def audit_pair(pair_id: str, variants: list[dict[str, Any]], findings: Findings) -> None:
    if len(variants) != 2:
        findings.check("pair has exactly two variants", False, f"{pair_id}: {len(variants)} variants")
        return
    findings.check("pair has exactly two variants", True, "")
    left, right = variants

    # evidence_removal deliberately takes the support away in one variant, so its
    # answer is meant to change. Every other family reorders or repeats the same
    # facts, where a changed answer would mean the pair is not a controlled test.
    if left["intervention"] != "evidence_removal":
        findings.check(
            "both variants agree on the gold state",
            left.get("gold_state") == right.get("gold_state"),
            f"{pair_id}: {left.get('gold_state')} vs {right.get('gold_state')}",
        )
    findings.check(
        "both variants stay in the same split",
        left.get("split") == right.get("split"),
        f"{pair_id}: {left.get('split')} vs {right.get('split')}",
    )
    findings.check(
        "both variants ask the same question",
        normalise(left.get("query", "")) == normalise(right.get("query", "")),
        f"{pair_id}: queries differ",
    )

    if left["intervention"] == "order_reversal":
        def content(row: dict[str, Any]) -> Counter[str]:
            return Counter(normalise(item.get("text", "")) for item in row.get("evidence") or [])

        findings.check(
            "order reversal presents identical evidence content",
            content(left) == content(right),
            f"{pair_id}: evidence content differs between variants",
        )

        def newest(row: dict[str, Any]) -> str | None:
            items = row.get("evidence") or []
            if not items:
                return None
            return max(items, key=lambda item: str(item.get("transaction_time") or ""))["hypothesis_id"]

        findings.check(
            "order reversal moves the newest arrival",
            newest(left) != newest(right),
            f"{pair_id}: newest arrival supports the same hypothesis in both variants",
        )


def write_report(findings: Findings, path: Path, rows: int, pairs: int) -> None:
    failed = {name: count for name, count in findings.counts.items() if count}
    lines = [
        "# Mechanical audit of the grounded benchmark",
        "",
        f"{rows} examples, {pairs} pairs. This audit checks only what can be settled",
        "without judgement: whether presented strings appear in the source record they",
        "cite, whether a pair's variants are the same facts reordered, and whether the",
        "gold state matches the construction. It records no approval.",
        "",
        f"**{len(failed)} of {len(findings.checked)} checks reported findings.**",
        "",
        "| Check | Checked | Findings |",
        "|---|---|---|",
    ]
    for name in sorted(findings.checked):
        lines.append(f"| {name} | {findings.checked[name]} | {findings.counts.get(name, 0)} |")
    lines.append("")

    lines += [
        "## How evidence prose relates to its source",
        "",
        "`verbatim` quotes the source record. `templated` generates the sentence but",
        "states a fact the record contains, so a reviewer can still check it against",
        "the source. `authored` has no anchor in the record at all and is the tier that",
        "needs a decision about what the paper claims.",
        "",
        "| Family | Verbatim | Templated | Authored |",
        "|---|---|---|---|",
    ]
    for family in sorted(findings.tiers):
        counts = findings.tiers[family]
        lines.append(
            f"| `{family}` | {counts['verbatim']} | {counts['templated']} | {counts['authored']} |"
        )
    totals = Counter()
    for counts in findings.tiers.values():
        totals.update(counts)
    grand = sum(totals.values()) or 1
    lines += [
        f"| **all** | **{totals['verbatim']} ({totals['verbatim'] / grand:.0%})** | "
        f"**{totals['templated']} ({totals['templated'] / grand:.0%})** | "
        f"**{totals['authored']} ({totals['authored'] / grand:.0%})** |",
        "",
    ]

    if failed:
        lines.append("## Findings")
        lines.append("")
        for name in sorted(failed):
            lines += [f"### {name} — {failed[name]}", ""]
            lines += [f"- `{detail}`" for detail in findings.examples[name]]
            if failed[name] > len(findings.examples[name]):
                lines.append(f"- … and {failed[name] - len(findings.examples[name])} more")
            lines.append("")
    else:
        lines += ["No mechanical findings. What remains is judgement:", "",
                  "whether each constructed example is a fair test of its intervention.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounded-dir", type=Path, default=Path("data/grounded"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = args.grounded_dir / f"river_bench_{split}.jsonl"
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    findings = Findings()
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        audit_row(row, findings)
        pairs[row["pair_id"]].append(row)
    for pair_id, variants in pairs.items():
        audit_pair(pair_id, sorted(variants, key=lambda item: item["example_id"]), findings)

    report = args.report or (args.grounded_dir / "audit_findings.md")
    write_report(findings, report, len(rows), len(pairs))

    for name in sorted(findings.checked):
        count = findings.counts.get(name, 0)
        status = "OK  " if count == 0 else "FAIL"
        print(f"{status} {name}: {count} findings / {findings.checked[name]} checked")
    print(f"\nreport -> {report}")


if __name__ == "__main__":
    main()
