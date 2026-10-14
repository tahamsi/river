"""JSON-stdin runner for the Corrective RAG (CRAG) native RIVER baseline.

Uses the retrieval evaluator of the pinned upstream checkout
(``external/CRAG``, commit de7c2961ae62): a ``T5ForSequenceClassification`` head
with ``num_labels=1`` that scores each (question, document) pair, thresholded
into CRAG's three actions exactly as ``scripts/CRAG_Inference.py`` does.

Upstream threshold convention, from ``process_flag`` and ``main``: the lower
threshold is negated before use, so with the repository's published defaults
(``--upper_threshold 0.592 --lower_threshold 0.995``) a document scores

* ``>= 0.592``   -> CORRECT
* ``>= -0.995``  -> AMBIGUOUS
* otherwise      -> INCORRECT

Mapping to RIVER states:

* CORRECT   -> ``active``   (retrieval is trusted and used directly)
* AMBIGUOUS -> ``disputed`` (CRAG combines internal and external knowledge
                             because the retrieved set is not decisive)
* INCORRECT -> ``abstain``  (CRAG discards the retrieval; with web search
                             disabled by RIVER's local-only rule there is no
                             corrective source, so nothing is asserted)

CRAG expresses no notion of temporal evolution, so this runner never emits
``evolution``.

The evaluator is trained by the upstream ``run_evaluator_training.sh`` and is not
a released weight. Without it there is no faithful CRAG, so this runner fails
with a setup message rather than substituting a simplified local scorer.

Environment:
    CRAG_EVALUATOR_PATH   directory holding the fine-tuned T5 evaluator (required)
    CRAG_UPPER_THRESHOLD  default 0.592
    CRAG_LOWER_THRESHOLD  default 0.995 (negated before comparison, per upstream)
    CRAG_DEVICE           default cuda when available, else cpu
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import river_runner_contract as contract

ACTION_CORRECT = "CORRECT"
ACTION_AMBIGUOUS = "AMBIGUOUS"
ACTION_INCORRECT = "INCORRECT"

ACTION_TO_STATE = {
    ACTION_CORRECT: "active",
    ACTION_AMBIGUOUS: "disputed",
    ACTION_INCORRECT: "abstain",
}


def _device() -> str:
    import torch

    configured = os.environ.get("CRAG_DEVICE")
    if configured:
        return configured
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _evaluator():
    """Load the upstream fine-tuned retrieval evaluator once per process."""

    from transformers import T5ForSequenceClassification, T5Tokenizer

    path = os.environ.get("CRAG_EVALUATOR_PATH")
    if not path or not os.path.isdir(path):
        raise contract.RunnerError(
            "CRAG_EVALUATOR_PATH is unset or not a directory. The retrieval "
            "evaluator is not a released weight: train it with "
            "external/CRAG/run_evaluator_training.sh and point CRAG_EVALUATOR_PATH "
            "at the result. This runner will not substitute a simplified scorer."
        )
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForSequenceClassification.from_pretrained(path, num_labels=1)
    model.to(_device())
    model.eval()
    return tokenizer, model


def _score(question: str, documents: list[str]) -> list[float]:
    """Relevance score per document, following the upstream inference loop."""

    import torch

    tokenizer, model = _evaluator()
    device = _device()
    scores: list[float] = []
    for document in documents:
        encoded = tokenizer(
            question,
            document,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        scores.append(float(logits.squeeze().item()))
    return scores


def _action(scores: list[float]) -> str:
    """CRAG's three-way action, matching upstream ``process_flag``."""

    upper = float(os.environ.get("CRAG_UPPER_THRESHOLD", "0.592"))
    lower = -float(os.environ.get("CRAG_LOWER_THRESHOLD", "0.995"))
    if any(score >= upper for score in scores):
        return ACTION_CORRECT
    if any(score >= lower for score in scores):
        return ACTION_AMBIGUOUS
    return ACTION_INCORRECT


def _generate(question: str, knowledge: list[str]) -> tuple[str, int | None, int | None]:
    """Answer from the accepted knowledge using the local open-weight model."""

    from openai import OpenAI

    settings = contract.local_llm()
    client = OpenAI(base_url=settings["base_url"], api_key=settings["api_key"])
    context = "\n".join(f"- {text}" for text in knowledge)
    completion = client.chat.completions.create(
        model=settings["model"],
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer the question using only the knowledge provided. "
                    "If the knowledge is insufficient, say so.\n\n"
                    f"Knowledge:\n{context}\n\nQuestion: {question}"
                ),
            }
        ],
    )
    usage = getattr(completion, "usage", None)
    return (
        (completion.choices[0].message.content or "").strip(),
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("query", "")).strip()
    documents = contract.evidence_text(payload)
    if not question:
        return contract.abstained("empty query", native_state="no_query")
    if not documents:
        return contract.abstained("no retrieved documents", native_state=ACTION_INCORRECT)

    scores = _score(question, documents)
    action = _action(scores)
    if action == ACTION_INCORRECT:
        # CRAG discards the retrieval and corrects via web search, which RIVER's
        # local-only inference rule disallows. Nothing is asserted.
        return contract.abstained("crag_discarded_retrieval", native_state=ACTION_INCORRECT)

    accepted = [
        document
        for document, score in zip(documents, scores)
        if score >= -float(os.environ.get("CRAG_LOWER_THRESHOLD", "0.995"))
    ] or documents
    answer, input_tokens, output_tokens = _generate(question, accepted)
    return contract.result(
        answer,
        predicted_state=ACTION_TO_STATE[action],
        citation_supported=action == ACTION_CORRECT,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        native_state=action,
    )


if __name__ == "__main__":
    contract.run(handle)
