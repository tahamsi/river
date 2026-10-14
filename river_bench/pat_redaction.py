"""Redact PAT-Questions text from release files and restore it from upstream.

PAT-Questions (github.com/jannatmeem95/PAT-Questions) publishes no licence, so
the public release does not redistribute its text. Every PAT question and
question template is replaced by a placeholder ``@@PAT:<sha256-prefix>:<form>@@``,
where ``form`` records how the string was serialised (``r`` raw, ``a`` JSON with
``\\uXXXX`` escapes). The answers, Wikidata identifiers and RIVER annotations stay;
they are facts or RIVER's own work, not PAT text.

Rehydration downloads the two March 2024 snapshot files at a pinned upstream
commit, checks their SHA-256, substitutes every placeholder back and checks the
restored file against the SHA-256 of the original. A file that does not restore
byte-for-byte is an error, never a silent difference.

``.tar.xz`` members are redacted inside the archive. The original tar header
blocks hold only paths and metadata, so they are kept verbatim and the archive is
reassembled around the restored members; with xz preset 6 the result matches the
original archive's SHA-256.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

PAT_REPOSITORY = "jannatmeem95/PAT-Questions"
PAT_COMMIT = "13f8fd00b9af53e6f97de8eae032adef47543fc0"
PAT_SNAPSHOT_FILES = {
    "singlehop": (
        "PAT-data/March2024/PAT-singlehop.json",
        "48412e990ea6f4c567fe9d4f777449c301056fb3a99029a7935fd47f20da3315",
    ),
    "multihop": (
        "PAT-data/March2024/PAT-multihop.json",
        "92f18a9de05507e46ca5968aef37999dbf39f5736726c981d538f128d27421b1",
    ),
}
MANIFEST_SCHEMA = "river-pat-redaction-v1"
XZ_PRESET = 6
PLACEHOLDER = re.compile(r"@@PAT:([0-9a-f]{16}):([ra])@@")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_snapshot(pat_dir: Path | None = None) -> dict[str, bytes]:
    """Return the pinned snapshot files, from ``pat_dir`` or from GitHub."""
    files = {}
    for task, (path, expected) in PAT_SNAPSHOT_FILES.items():
        if pat_dir is not None:
            payload = (pat_dir / path).read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/{PAT_REPOSITORY}/{PAT_COMMIT}/{path}"
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
        actual = sha256_bytes(payload)
        if actual != expected:
            raise ValueError(f"{path}: SHA-256 {actual} does not match the pinned {expected}")
        files[task] = payload
    return files


def pat_strings(snapshot: dict[str, bytes]) -> set[str]:
    """Every question and template in the snapshot, including mismatched dict keys."""
    strings: set[str] = set()
    for payload in snapshot.values():
        for key, record in json.loads(payload).items():
            strings.update({key, record["question"], record["template"]})
    return strings


def _key(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))[:16]


def _forms(text: str) -> dict[str, str]:
    forms = {"r": text}
    escaped = json.dumps(text, ensure_ascii=True)[1:-1]
    if escaped != json.dumps(text, ensure_ascii=False)[1:-1]:
        forms["a"] = escaped
    return forms


def _trie_pattern(words: list[str]) -> str:
    """A regex equivalent to ``a|b|...`` that the matcher walks as a trie."""
    trie: dict[str, Any] = {}
    for word in words:
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node[""] = True

    def render(node: dict[str, Any]) -> str:
        branches = [re.escape(char) + render(child) for char, child in sorted(node.items()) if char]
        optional = "" in node
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        if optional:
            body = "(?:" + body + ")?"
        return body

    return render(trie)


class Redactor:
    def __init__(self, strings: set[str]):
        self.lookup: dict[str, str] = {}
        variants: dict[str, str] = {}
        for text in strings:
            key = _key(text)
            if self.lookup.setdefault(key, text) != text:
                raise ValueError(f"placeholder key collision: {key}")
            for form, variant in _forms(text).items():
                variants[variant] = f"@@PAT:{key}:{form}@@"
        self.variants = variants
        # Greedy trie matching takes the longest string at each position.
        self.pattern = re.compile(_trie_pattern(list(variants)))

    def redact(self, text: str) -> tuple[str, int]:
        if PLACEHOLDER.search(text):
            raise ValueError("input already contains a PAT placeholder")
        return self.pattern.subn(lambda match: self.variants[match.group(0)], text)

    def rehydrate(self, text: str) -> str:
        def restore(match: re.Match[str]) -> str:
            text = self.lookup.get(match.group(1))
            if text is None:
                raise ValueError(f"unknown PAT placeholder {match.group(0)}")
            return _forms(text)[match.group(2)]

        return PLACEHOLDER.sub(restore, text)

    def leaks(self, text: str) -> int:
        return len(self.pattern.findall(text))


def _redact_bytes(redactor: Redactor, payload: bytes) -> tuple[bytes, int]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload, 0
    redacted, count = redactor.redact(text)
    return redacted.encode("utf-8"), count


def _tar_layout(archive: bytes) -> tuple[bytes, list[tuple[tarfile.TarInfo, bytes, bytes]], bytes]:
    tar_bytes = lzma.decompress(archive)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as handle:
        members = handle.getmembers()
    layout = []
    end = 0
    for member in members:
        header = tar_bytes[member.offset : member.offset_data]
        data = tar_bytes[member.offset_data : member.offset_data + member.size] if member.isfile() else b""
        layout.append((member, header, data))
        end = member.offset_data + ((member.size + 511) // 512) * 512 if member.isfile() else member.offset_data
    return tar_bytes, layout, tar_bytes[end:]


def redact_file(redactor: Redactor, source: Path, target: Path, relative: str) -> dict[str, Any] | None:
    """Write a redacted copy of ``source`` to ``target``; return its manifest entry if PAT text was found."""
    original = source.read_bytes()
    if relative.endswith(".tar.xz"):
        tar_bytes, layout, trailer = _tar_layout(original)
        if lzma.compress(tar_bytes, preset=XZ_PRESET) != original:
            raise ValueError(f"{relative}: xz preset {XZ_PRESET} does not reproduce the archive")
        entries = []
        total = 0
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as handle:
            for member, header, data in layout:
                redacted, count = _redact_bytes(redactor, data) if member.isfile() else (data, 0)
                total += count
                entries.append(
                    {
                        "name": member.name,
                        "header": base64.b64encode(header).decode("ascii"),
                        "sha256": sha256_bytes(data) if member.isfile() else None,
                        "redacted": count > 0,
                    }
                )
                copy = tarfile.TarInfo(member.name)
                copy.type, copy.mode, copy.mtime = member.type, member.mode, member.mtime
                copy.size = len(redacted) if member.isfile() else 0
                handle.addfile(copy, io.BytesIO(redacted) if member.isfile() else None)
        if not total:
            return None
        target.write_bytes(lzma.compress(output.getvalue(), preset=XZ_PRESET))
        return {
            "path": relative,
            "kind": "tar.xz",
            "sha256": sha256_bytes(original),
            "replacements": total,
            "members": entries,
            "trailer_bytes": len(trailer),
        }
    redacted, count = _redact_bytes(redactor, original)
    if not count:
        return None
    target.write_bytes(redacted)
    return {"path": relative, "kind": "text", "sha256": sha256_bytes(original), "replacements": count}


def rehydrate_entry(redactor: Redactor, root: Path, entry: dict[str, Any]) -> bytes:
    """Rebuild one original file from its redacted copy; raise unless the SHA-256 matches."""
    path = root / entry["path"]
    current = path.read_bytes()
    if sha256_bytes(current) == entry["sha256"]:
        return current
    if entry["kind"] == "text":
        restored = redactor.rehydrate(current.decode("utf-8")).encode("utf-8")
    else:
        with tarfile.open(fileobj=io.BytesIO(lzma.decompress(current))) as handle:
            payloads = {m.name: handle.extractfile(m).read() for m in handle.getmembers() if m.isfile()}
        chunks = []
        for member in entry["members"]:
            chunks.append(base64.b64decode(member["header"]))
            if member["sha256"] is None:
                continue
            data = payloads[member["name"]]
            if member["redacted"]:
                data = redactor.rehydrate(data.decode("utf-8")).encode("utf-8")
            if sha256_bytes(data) != member["sha256"]:
                raise ValueError(f"{entry['path']}::{member['name']}: restored member does not match")
            chunks.append(data + b"\0" * (-len(data) % 512))
        chunks.append(b"\0" * entry["trailer_bytes"])
        restored = lzma.compress(b"".join(chunks), preset=XZ_PRESET)
    if sha256_bytes(restored) != entry["sha256"]:
        raise ValueError(f"{entry['path']}: restored file does not match its recorded SHA-256")
    return restored
