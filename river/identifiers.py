"""Benchmark identifiers that say nothing about the answer.

The corpus builders name hypotheses and evidence by their role in the
construction (``h_gold`` and ``h_alt``, ``h_supported`` and
``h_unsupported_late``, ``h_current`` and ``h_expired``) and list the gold
hypothesis in the same position for every row of a family. Any system that sees
an identifier, or breaks a tie by list position, can read the label without
reading any evidence. The answer prompt printed hypothesis ids, and naive RAG's
answer accuracy fell from 0.876 to 0.684 once they were hashed (see
docs/integrity.md).

``anonymise_row`` replaces every hypothesis and evidence id with a hash of the
original id, and orders hypotheses by that hash. The hash depends only on the
original id, so an id shared by the two members of a pair, or by a corpus and
one derived from it, maps to the same value everywhere and paired metrics still
line up. Identifiers already in hashed form are left alone, so the transform is
idempotent. Evidence order is arrival order, which the benchmark varies on
purpose, and is not touched; nor are text, times or verifier scores.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

NAMESPACE = "river-bench-id-v1"
HASHED = re.compile(r"^[hec]_[0-9a-f]{12}$")
# Role words the builders have used. Any that survive the id mapping (inside a
# source id or a chunk id, for example) are rewritten too, and a test fails if one is
# ever written to disk.
ROLE_NAMES = (
    "gold", "alt", "supported", "unsupported_late", "repeated_weak", "current",
    "expired", "new", "old", "original", "corrected", "claim", "left", "right",
    "late", "phase_one", "phase_two", "insufficient_context",
)
ROLE_ID = re.compile(r":(h|e)_(" + "|".join(ROLE_NAMES) + r")\b")
UNTOUCHED_FIELDS = {"text", "query"}


def _digest(value: str) -> str:
    return hashlib.sha1(f"{NAMESPACE}|{value}".encode("utf-8")).hexdigest()[:12]


def hashed_id(original: str, kind: str) -> str:
    """``prefix:h_gold`` -> ``prefix:h_<hash>``; ids already hashed pass through."""

    prefix, _, local = original.rpartition(":")
    if HASHED.match(local):
        return original
    new_local = f"{kind}_{_digest(original)}"
    return f"{prefix}:{new_local}" if prefix else new_local


def anonymise_row(row: dict[str, Any]) -> dict[str, Any]:
    mapping: dict[str, str] = {}
    for hypothesis in row.get("hypotheses") or []:
        if isinstance(hypothesis, dict) and hypothesis.get("hypothesis_id"):
            mapping[hypothesis["hypothesis_id"]] = hashed_id(hypothesis["hypothesis_id"], "h")
    for evidence in row.get("evidence") or []:
        if isinstance(evidence, dict):
            if evidence.get("evidence_id"):
                mapping[evidence["evidence_id"]] = hashed_id(evidence["evidence_id"], "e")
            if evidence.get("hypothesis_id") and evidence["hypothesis_id"] not in mapping:
                mapping[evidence["hypothesis_id"]] = hashed_id(evidence["hypothesis_id"], "h")
    for key in ("gold_hypothesis_id", "intervention_hypothesis_id"):
        value = row.get(key)
        if isinstance(value, str) and value and value not in mapping:
            mapping[value] = hashed_id(value, "h")
    mapping = {old: new for old, new in mapping.items() if old != new}

    # Longest first, so an evidence id is replaced before a shorter id it contains.
    pattern = re.compile("|".join(re.escape(old) for old in sorted(mapping, key=len, reverse=True))) if mapping else None

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            if pattern is not None:
                value = pattern.sub(lambda match: mapping[match.group(0)], value)
            return ROLE_ID.sub(lambda match: f":{match.group(1)}_{_digest(match.group(0) + '|' + value)}", value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {rewrite(k): (v if k in UNTOUCHED_FIELDS else rewrite(v)) for k, v in value.items()}
        return value

    result = {key: (value if key in UNTOUCHED_FIELDS else rewrite(value)) for key, value in row.items()}
    if isinstance(result.get("hypotheses"), list):
        result["hypotheses"] = sorted(
            result["hypotheses"],
            key=lambda hypothesis: str(hypothesis.get("hypothesis_id", "")) if isinstance(hypothesis, dict) else "",
        )
    return result


def role_named_ids(row: dict[str, Any]) -> list[str]:
    """Every role-named identifier still present anywhere in a row."""

    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            found.extend(match.group(0) for match in ROLE_ID.finditer(value))
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(key)
                if key not in UNTOUCHED_FIELDS:
                    walk(item)

    walk(row)
    return found
