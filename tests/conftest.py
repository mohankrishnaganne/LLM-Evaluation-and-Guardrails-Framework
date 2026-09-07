"""Shared fixtures and test doubles."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from llm_eval_guardrails.config import (
    AWSSettings,
    EvaluationSettings,
    GuardrailSettings,
    LLMSettings,
    Settings,
)
from llm_eval_guardrails.llm.base import ChatMessage, LLMClient, LLMResponse, TokenUsage
from llm_eval_guardrails.models import RAGSample, RetrievedDocument


class FakeLLMClient(LLMClient):
    """A scripted LLM client that records calls and returns canned responses.

    Responses are consumed in order; once exhausted, the last one repeats so a
    test does not have to script every call a metric happens to make.
    """

    provider_name = "fake"

    def __init__(
        self,
        responses: Sequence[str | Exception] | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        super().__init__(settings or LLMSettings(max_attempts=1, requests_per_minute=None))
        self.responses: list[str | Exception] = list(responses or ['{"score": 1.0}'])
        self.prompts: list[str] = []
        self.systems: list[str | None] = []
        self.temperatures: list[float] = []
        self._index = 0

    async def _invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
        model_id: str,
    ) -> LLMResponse:
        self.prompts.append(messages[-1].content)
        self.systems.append(system)
        self.temperatures.append(temperature)

        index = min(self._index, len(self.responses) - 1)
        self._index += 1
        item = self.responses[index]
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            text=item,
            model_id=model_id,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_ms=1.0,
            stop_reason="end_turn",
        )


def claims_response(*verdicts: str) -> str:
    """Build a claim-list judge payload with the given verdicts.

    Args:
        *verdicts: One verdict string per synthesised claim.

    Returns:
        The JSON payload.
    """
    return json.dumps(
        {
            "claims": [
                {
                    "claim": f"claim {i}",
                    "verdict": verdict,
                    "supporting_context_ids": [1] if verdict == "supported" else [],
                    "reasoning": "because",
                }
                for i, verdict in enumerate(verdicts, start=1)
            ]
        }
    )


def statements_response(*flags: bool) -> str:
    """Build a boolean-statement judge payload.

    Args:
        *flags: One boolean per statement.

    Returns:
        The JSON payload.
    """
    return json.dumps({"statements": list(flags), "reasoning": "because"})


@pytest.fixture
def llm_settings() -> LLMSettings:
    return LLMSettings(
        model_id="test-model",
        max_attempts=1,
        requests_per_minute=None,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.002,
    )


@pytest.fixture
def guardrail_settings() -> GuardrailSettings:
    return GuardrailSettings()


@pytest.fixture
def eval_settings() -> EvaluationSettings:
    return EvaluationSettings(max_concurrency=2, consistency_samples=2)


@pytest.fixture
def settings(
    llm_settings: LLMSettings,
    guardrail_settings: GuardrailSettings,
    eval_settings: EvaluationSettings,
) -> Settings:
    return Settings(
        llm=llm_settings,
        aws=AWSSettings(region="us-east-1", report_bucket="test-bucket"),
        evaluation=eval_settings,
        guardrails=guardrail_settings,
    )


@pytest.fixture
def fake_llm(llm_settings: LLMSettings) -> FakeLLMClient:
    return FakeLLMClient(settings=llm_settings)


@pytest.fixture
def sample() -> RAGSample:
    return RAGSample(
        sample_id="s-1",
        question="What is the refund window?",
        answer="You can request a refund within 30 days of purchase.",
        contexts=[
            RetrievedDocument(
                doc_id="policy-4",
                content="Refunds are accepted within 30 days of purchase.",
                rank=0,
            ),
            RetrievedDocument(doc_id="policy-9", content="Shipping takes 5 days.", rank=1),
        ],
        ground_truth="Refunds are available for 30 days.",
    )


@pytest.fixture
def jsonl_dataset(tmp_path: Any) -> str:
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "sample_id": f"s-{i}",
            "question": f"Question {i}?",
            "answer": f"Answer {i}.",
            "contexts": [f"Context for question {i}."],
            "ground_truth": f"Answer {i}.",
        }
        for i in range(3)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return str(path)
