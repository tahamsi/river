"""Strictly score the explicit direct answer emitted under the decision-render generation contract.

An earlier evaluator searched the whole response for the gold object. It could
therefore count a wrong first sentence as correct when the explanation merely
mentioned the gold entity later. This scorer reads only a mandatory first-line
``FINAL_ANSWER:`` field and compares that field exactly, after conservative
typographic normalization, with the gold answer and recorded aliases.

Prediction files are never overwritten unless ``--in-place`` is explicitly
supplied. Release runs must use ``--output-dir`` and retain the unscored source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


FINAL_ANSWER_RE = re.compile(r"^\s*FINAL_ANSWER\s*:\s*(.*?)\s*$", re.IGNORECASE)
NON_ACTIVE_PREFIXES = {
    "abstain": "ABSTAIN:",
    "unknown": "UNKNOWN:",
    "disputed": "DISPUTED:",
    "evolution": "EVOLUTION:",
}


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", str(text or ""))
    folded = folded.replace("’", "'").replace("“", '"').replace("”", '"')
    folded = re.sub(r"[`*_]", "", folded)
    folded = re.sub(r"\s+", " ", folded)
    return folded.strip().strip(" \t\r\n.,;:!?\"'").casefold()


def parse_final_answer(answer: str | None) -> str | None:
    if not answer:
        return None
    for line in str(answer).splitlines():
        if not line.strip():
            continue
        match = FINAL_ANSWER_RE.fullmatch(line)
        return match.group(1).strip() if match and match.group(1).strip() else None
    return None


def _transformation_metadata(item: dict) -> dict:
    metadata = item.get("transformation_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _hypotheses(item: dict) -> list[dict]:
    return [row for row in item.get("hypotheses", []) if isinstance(row, dict)]


def _answer_candidates(item: dict, hypothesis_id: str | None) -> set[str]:
    values: list[str] = []
    metadata = _transformation_metadata(item)
    if hypothesis_id == item.get("gold_hypothesis_id"):
        if metadata.get("gold_answer") is not None:
            values.append(str(metadata["gold_answer"]))
        aliases = metadata.get("gold_answer_aliases") or []
        values.extend(str(value) for value in aliases if value is not None)
    for row in _hypotheses(item):
        if row.get("hypothesis_id") == hypothesis_id and row.get("object") is not None:
            values.append(str(row["object"]))
    return {normalise(value) for value in values if normalise(value)}


def named_hypothesis(item: dict, direct_answer: str | None) -> str | None:
    value = normalise(direct_answer or "")
    if not value:
        return None
    matches = [
        row.get("hypothesis_id")
        for row in _hypotheses(item)
        if value in _answer_candidates(item, row.get("hypothesis_id"))
    ]
    matches = [value for value in matches if value]
    return matches[0] if len(matches) == 1 else None


def score(item: dict, answer: str | None) -> bool | None:
    """Strict direct-answer accuracy, or None where no single answer is gold."""

    gold_id = item.get("gold_hypothesis_id")
    if gold_id is None or str(item.get("gold_state") or "").lower() != "active":
        return None
    direct = parse_final_answer(answer)
    if direct is None:
        return False
    return normalise(direct) in _answer_candidates(item, gold_id)


def contract_valid(row: dict, answer: str | None) -> bool:
    state = str(row.get("predicted_state") or "").lower()
    if state in NON_ACTIVE_PREFIXES:
        return bool(answer) and str(answer).strip().upper().startswith(NON_ACTIVE_PREFIXES[state])
    return parse_final_answer(answer) is not None


def decision_faithful(item: dict, row: dict, direct_answer: str | None) -> bool | None:
    selected = row.get("predicted_hypothesis_id")
    if selected is None:
        return None
    if direct_answer is None:
        return False
    return normalise(direct_answer) in _answer_candidates(item, selected)


def end_to_end_correct(item: dict, row: dict, text_correct: bool | None, faithful: bool | None) -> bool | None:
    if text_correct is None or row.get("predicted_state") is None:
        return None
    return bool(
        str(row.get("predicted_state")).lower() == "active"
        and row.get("predicted_hypothesis_id") == item.get("gold_hypothesis_id")
        and text_correct
        and faithful
    )


def load_fixture(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                index[item["example_id"]] = item
    return index


def score_file(path: Path, destination: Path, fixture: dict[str, dict]) -> tuple[int, int, int, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = correct = unmatched = violations = 0
    for row in rows:
        item = fixture.get(row.get("example_id"))
        if item is None:
            unmatched += 1
            row.update(
                answer_text_correct=None,
                answer_named_hypothesis_id=None,
                answer_contract_valid=False,
                direct_answer=None,
                answer_decision_faithful=None,
                end_to_end_answer_correct=None,
            )
            continue
        direct = parse_final_answer(row.get("answer"))
        verdict = score(item, row.get("answer"))
        faithful = decision_faithful(item, row, direct)
        valid = contract_valid(row, row.get("answer"))
        row["direct_answer"] = direct
        row["answer_text_correct"] = verdict
        row["answer_named_hypothesis_id"] = named_hypothesis(item, direct)
        row["answer_contract_valid"] = valid
        row["answer_decision_faithful"] = faithful
        row["end_to_end_answer_correct"] = end_to_end_correct(item, row, verdict, faithful)
        if verdict is not None:
            scored += 1
            correct += int(verdict)
        if not valid:
            violations += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(destination)
    return scored, correct, unmatched, violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--allow-contract-violations",
        action="store_true",
        help="Diagnostic only; release scoring fails when the rendering contract is violated.",
    )
    args = parser.parse_args()

    fixture = load_fixture(args.fixture)
    exit_code = 0
    for path in args.predictions:
        target = path if args.in_place else args.output_dir / path.name
        scored, correct, unmatched, violations = score_file(path, target, fixture)
        rate = f"{correct / scored:.3f}" if scored else "n/a"
        print(
            f"{path.name}: {correct}/{scored} strict direct answers correct ({rate}); "
            f"{violations} contract violations; {unmatched} unmatched rows"
        )
        if unmatched or (violations and not args.allow_contract_violations):
            exit_code = 2
    if exit_code:
        print("Strict answer scoring failed its release gate.", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
