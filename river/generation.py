"""Local, open-weight text generation backends used by controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from openai import DefaultHttpxClient, OpenAI


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LocalOpenAICompatibleGenerator:
    """Call an OpenAI-compatible server running on the local machine.

    Ollama, vLLM and llama.cpp all expose this protocol. No hosted API key is
    required; the placeholder key exists only because the client validates it.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434/v1",
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 120.0,
        seed: int | None = None,
    ):
        if not model.strip():
            raise ValueError("local model name must not be empty")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("local generation endpoint must resolve explicitly to localhost")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        # A loopback-only model must not inherit HTTP(S)/SOCKS proxy settings
        # from the host. Besides breaking offline runs, a proxy would violate
        # the promise that generation stays on the local server.
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="local-open-weight",
            timeout=timeout,
            http_client=DefaultHttpxClient(trust_env=False),
        )

    def generate(self, query: str, context: str, *, instruction: str | None = None) -> GenerationResult:
        system = instruction or (
            "Answer only from the supplied evidence. If it is insufficient, say so. "
            "Do not use outside knowledge. Keep the answer concise and cite chunk IDs in square brackets."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question:\n{query}\n\nEvidence:\n{context}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            extra_body={"reasoning_effort": "none"},
        )
        usage = response.usage
        text = response.choices[0].message.content or ""
        if not text.strip():
            raise RuntimeError(
                f"local model {self.model!r} returned no final answer; check thinking/reasoning settings "
                "or increase --max-output-tokens"
            )
        return GenerationResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )

    def __call__(self, query: str, context: str) -> str:
        return self.generate(query, context).text

    def manifest(self) -> dict[str, Any]:
        return {
            "provider": "local_openai_compatible",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "open_weight": True,
            "remote_paid_api": False,
        }
