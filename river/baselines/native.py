"""Native external baseline manifests and command runners."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from river.schemas import ComparisonMode

from .base import Baseline, BaselineCapabilities, BaselineUnavailable


MANIFEST_DIR = Path(__file__).resolve().parents[2] / "configs" / "native_baselines"


@dataclass(frozen=True)
class NativeBaselineManifest:
    baseline_id: str
    display_name: str
    official_repository_url: str
    commit_or_tag: str
    paper_reference: str
    license: str
    installation_command: str
    isolation: str
    model_retriever_requirements: list[str]
    required_credentials: list[str]
    supported_execution_modes: list[str]
    expected_output_mapping: dict[str, Any]
    default_command: list[str]
    readiness_notes: str
    state_support: dict[str, str]

    @classmethod
    def load(cls, baseline_id: str) -> "NativeBaselineManifest":
        path = MANIFEST_DIR / f"{baseline_id}.json"
        if not path.exists():
            raise BaselineUnavailable(f"No native baseline manifest found for {baseline_id!r} at {path}")
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "display_name": self.display_name,
            "official_repository_url": self.official_repository_url,
            "commit_or_tag": self.commit_or_tag,
            "paper_reference": self.paper_reference,
            "license": self.license,
            "installation_command": self.installation_command,
            "isolation": self.isolation,
            "model_retriever_requirements": self.model_retriever_requirements,
            "required_credentials": self.required_credentials,
            "supported_execution_modes": self.supported_execution_modes,
            "expected_output_mapping": self.expected_output_mapping,
            "default_command": self.default_command,
            "readiness_notes": self.readiness_notes,
            "state_support": self.state_support,
        }


def load_native_manifests() -> dict[str, NativeBaselineManifest]:
    manifests: dict[str, NativeBaselineManifest] = {}
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests[payload["baseline_id"]] = NativeBaselineManifest(**payload)
    return manifests


def _normalize_command(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    return [str(item) for item in value]


def _credential_status(names: list[str]) -> dict[str, bool]:
    return {name: bool(os.environ.get(name)) for name in names}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _looks_like_commit(value: str) -> bool:
    return len(value) == 40 and all(char in "0123456789abcdef" for char in value.lower())


class NativeCommandAdapter(Baseline):
    """Runs an official external system through an isolated command contract."""

    baseline_id = ""
    capabilities = BaselineCapabilities(native_external_system=True)

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.native_manifest = NativeBaselineManifest.load(self.baseline_id or self.name)
        self.comparison_mode = ComparisonMode(self.config.get("comparison_mode", ComparisonMode.CONTROLLED.value))
        if self.comparison_mode.value not in self.native_manifest.supported_execution_modes:
            raise BaselineUnavailable(
                f"{self.name} does not support comparison_mode={self.comparison_mode.value!r}; "
                f"supported modes: {self.native_manifest.supported_execution_modes}"
            )
        self.timeout_seconds = float(self.config.get("timeout_seconds", 120.0))
        self.max_retries = int(self.config.get("max_retries", 0))
        self.rate_limit_per_minute = self.config.get("rate_limit_per_minute")
        self.cost_per_input_token = float(self.config.get("cost_per_input_token", 0.0))
        self.cost_per_output_token = float(self.config.get("cost_per_output_token", 0.0))

    @property
    def version(self) -> str:
        return self.native_manifest.commit_or_tag[:12]

    def manifest_for_run(self) -> dict[str, Any]:
        data = self.native_manifest.as_dict()
        data["comparison_mode"] = self.comparison_mode.value
        return data

    def process_turn(self, session_id, query, retrieved_chunks, target_time=None):
        working_dir = self._validated_checkout_path() or self.config.get("working_dir")
        command = _normalize_command(self.config.get("runner_command")) or self.native_manifest.default_command
        if not command:
            raise BaselineUnavailable(self._setup_message("no runner_command configured"))

        missing_credentials = [
            name for name, present in _credential_status(self.native_manifest.required_credentials).items() if not present
        ]
        if missing_credentials and not self.config.get("allow_missing_credentials", False):
            raise BaselineUnavailable(self._setup_message(f"missing credentials: {', '.join(missing_credentials)}"))

        payload = {
            "baseline_id": self.name,
            "comparison_mode": self.comparison_mode.value,
            "session_id": session_id,
            "query": query,
            "target_time": target_time,
            "evidence": retrieved_chunks,
            "budgets": self.config.get("budgets", {}),
            "native_config": self.config.get("native_config", {}),
        }
        start = time.perf_counter()
        attempts = 0
        last_error = ""
        while attempts <= self.max_retries:
            attempts += 1
            if self.rate_limit_per_minute:
                time.sleep(60.0 / float(self.rate_limit_per_minute))
            try:
                completed = subprocess.run(
                    command,
                    input=_json_dumps(payload),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                    cwd=working_dir,
                )
            except subprocess.TimeoutExpired as exc:
                last_error = f"timeout after {self.timeout_seconds}s: {exc}"
                continue
            if completed.returncode == 0:
                try:
                    raw = json.loads(completed.stdout or "{}")
                except json.JSONDecodeError as exc:
                    last_error = f"runner emitted non-JSON stdout: {exc}; stderr={completed.stderr[:400]}"
                    continue
                return self._normalize_output(raw, attempts=attempts, elapsed_ms=(time.perf_counter() - start) * 1000)
            last_error = f"runner exited {completed.returncode}: {completed.stderr[:800]}"
        raise BaselineUnavailable(self._setup_message(last_error))

    def _validated_checkout_path(self) -> str | None:
        checkout = self.config.get("official_checkout_path") or self.config.get("checkout_path")
        if not checkout:
            return None
        checkout_path = Path(checkout)
        commit = subprocess.run(
            ["git", "-C", str(checkout_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode != 0:
            raise BaselineUnavailable(self._setup_message(f"checkout_path is not a Git repository: {checkout_path}"))
        expected = self.native_manifest.commit_or_tag
        current = commit.stdout.strip()
        if _looks_like_commit(expected) and current != expected:
            raise BaselineUnavailable(
                self._setup_message(f"checkout_path must be at {expected}; found {current}")
            )
        return str(checkout_path)

    def _setup_message(self, reason: str) -> str:
        return (
            f"{self.name} is not ready: {reason}. Install/configure the official implementation from "
            f"{self.native_manifest.official_repository_url} at {self.native_manifest.commit_or_tag}. "
            f"Install command: {self.native_manifest.installation_command}. "
            f"Required credentials: {self.native_manifest.required_credentials or 'none'}."
        )

    def _normalize_output(self, raw: dict[str, Any], *, attempts: int, elapsed_ms: float) -> dict[str, Any]:
        answer = str(raw.get("answer", raw.get("final_answer", "")))
        emitted_state = raw.get("predicted_state") or raw.get("state")
        state_source = "native" if emitted_state else "unsupported"
        predicted_state = str(emitted_state).lower() if emitted_state else None
        if predicted_state and predicted_state not in {"active", "disputed", "evolution", "unknown", "abstain"}:
            predicted_state = "unknown"
            state_source = "evaluator_derived"
        input_tokens = raw.get("input_tokens")
        output_tokens = raw.get("output_tokens")
        cost_usd = raw.get("cost_usd")
        if cost_usd is None:
            cost_usd = (
                (float(input_tokens or 0) * self.cost_per_input_token)
                + (float(output_tokens or 0) * self.cost_per_output_token)
            )
        return {
            "answer": answer,
            "predicted_state": predicted_state,
            "predicted_hypothesis_id": raw.get("predicted_hypothesis_id"),
            "confidence": raw.get("confidence"),
            "answer_correct": raw.get("answer_correct"),
            "citation_supported": raw.get("citation_supported"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "comparison_mode": self.comparison_mode.value,
            "native_execution": {
                "attempts": attempts,
                "elapsed_ms": elapsed_ms,
                "timeout_seconds": self.timeout_seconds,
                "rate_limit_per_minute": self.rate_limit_per_minute,
                "cost_usd": cost_usd,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "state_source": state_source,
                "state_support": self.native_manifest.state_support,
                "raw_output": raw,
                "manifest": self.manifest_for_run(),
            },
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": asdict(self.capabilities),
            "config": {key: value for key, value in self.config.items() if key != "runner_command"},
            "native_manifest": self.manifest_for_run(),
        }


class SelfRAGAdapter(NativeCommandAdapter):
    name = baseline_id = "self_rag"


class CRAGAdapter(NativeCommandAdapter):
    name = baseline_id = "crag"


class GraphRAGLocalAdapter(NativeCommandAdapter):
    name = baseline_id = "graphrag_local"


class GraphRAGGlobalAdapter(NativeCommandAdapter):
    name = baseline_id = "graphrag_global"


class GraphRAGAdapter(GraphRAGLocalAdapter):
    name = "graphrag"


class MemGPTAdapter(NativeCommandAdapter):
    name = baseline_id = "memgpt"


class ZepAdapter(NativeCommandAdapter):
    name = baseline_id = "zep"

