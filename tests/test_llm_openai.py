"""Tests for the OpenAI provider, ensuring parity with the Bedrock client.

These require the ``openai`` extra, which the dev environment installs. They
use stub response objects rather than the real SDK client, so no network access
or credentials are needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

openai = pytest.importorskip("openai")

from llm_eval_guardrails.config import LLMProvider, LLMSettings, Settings  # noqa: E402
from llm_eval_guardrails.exceptions import (  # noqa: E402
    LLMError,
    LLMRateLimitError,
    LLMTransientError,
)
from llm_eval_guardrails.llm.base import ChatMessage  # noqa: E402
from llm_eval_guardrails.llm.openai_client import OpenAILLMClient  # noqa: E402
from llm_eval_guardrails.llm.registry import build_llm_client  # noqa: E402


def completion(text: str = "hello", finish_reason: str = "stop") -> SimpleNamespace:
    """Build a stub resembling an OpenAI chat completion."""
    return SimpleNamespace(
        id="cmpl-1",
        model="gpt-test",
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
    )


class StubOpenAI:
    """Minimal stand-in for ``AsyncOpenAI``."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.requests: list[dict] = []
        self._result = result if result is not None else completion()
        self._error = error
        self.closed = False

        outer = self

        class Completions:
            @staticmethod
            async def create(**request):
                outer.requests.append(request)
                if outer._error is not None:
                    raise outer._error
                return outer._result

        self.chat = SimpleNamespace(completions=Completions())

    async def close(self) -> None:
        self.closed = True


def settings() -> LLMSettings:
    return LLMSettings(
        provider=LLMProvider.OPENAI,
        model_id="gpt-test",
        max_attempts=1,
        requests_per_minute=None,
    )


def make_client(stub: StubOpenAI, config: LLMSettings | None = None) -> OpenAILLMClient:
    # The stub models only the slice of AsyncOpenAI the client actually calls,
    # so the cast is the honest way to express "duck-typed test double".
    return OpenAILLMClient(config or settings(), client=cast(Any, stub))


def api_error(status: int) -> Any:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=status, request=request, json={"error": {}})
    return openai.APIStatusError("boom", response=response, body=None)


class TestOpenAIClient:
    async def test_parses_a_completion(self):
        stub = StubOpenAI()
        client = make_client(stub)
        response = await client.acomplete([ChatMessage.user("hi")])
        assert response.text == "hello"
        assert response.usage.input_tokens == 11
        assert response.usage.output_tokens == 4
        assert response.stop_reason == "stop"

    async def test_prepends_the_system_message(self):
        stub = StubOpenAI()
        client = make_client(stub)
        await client.acomplete([ChatMessage.user("hi")], system="be terse")
        messages = stub.requests[0]["messages"]
        assert messages[0] == {"role": "system", "content": "be terse"}
        assert messages[1] == {"role": "user", "content": "hi"}

    async def test_omits_the_system_message_when_absent(self):
        stub = StubOpenAI()
        await make_client(stub).acomplete([ChatMessage.user("hi")])
        assert stub.requests[0]["messages"][0]["role"] == "user"

    async def test_passes_sampling_parameters(self):
        stub = StubOpenAI()
        config = settings()
        config.top_p = 0.9
        await make_client(stub, config).acomplete(
            [ChatMessage.user("hi")], temperature=0.3, max_tokens=64
        )
        request = stub.requests[0]
        assert request["temperature"] == 0.3
        assert request["max_tokens"] == 64
        assert request["top_p"] == 0.9

    async def test_empty_completion_is_transient(self):
        stub = StubOpenAI(result=completion(text=""))
        client = make_client(stub)
        with pytest.raises(LLMTransientError, match="empty completion"):
            await client.acomplete([ChatMessage.user("hi")])

    async def test_timeout_is_transient(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        stub = StubOpenAI(error=openai.APITimeoutError(request=request))
        client = make_client(stub)
        with pytest.raises(LLMTransientError):
            await client.acomplete([ChatMessage.user("hi")])

    async def test_server_error_is_transient(self):
        stub = StubOpenAI(error=api_error(503))
        client = make_client(stub)
        with pytest.raises(LLMTransientError):
            await client.acomplete([ChatMessage.user("hi")])

    async def test_client_error_is_permanent(self):
        stub = StubOpenAI(error=api_error(400))
        client = make_client(stub)
        with pytest.raises(LLMError) as excinfo:
            await client.acomplete([ChatMessage.user("hi")])
        assert not isinstance(excinfo.value, LLMTransientError)

    async def test_rate_limit_maps_to_rate_limit_error(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(
            status_code=429,
            request=request,
            headers={"retry-after": "2"},
            json={"error": {}},
        )
        stub = StubOpenAI(error=openai.RateLimitError("slow down", response=response, body=None))
        client = make_client(stub)
        with pytest.raises(LLMRateLimitError) as excinfo:
            await client.acomplete([ChatMessage.user("hi")])
        assert excinfo.value.retry_after == 2.0

    async def test_missing_retry_after_header_is_tolerated(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(status_code=429, request=request, json={"error": {}})
        stub = StubOpenAI(error=openai.RateLimitError("slow down", response=response, body=None))
        client = make_client(stub)
        with pytest.raises(LLMRateLimitError) as excinfo:
            await client.acomplete([ChatMessage.user("hi")])
        assert excinfo.value.retry_after is None

    async def test_aclose_closes_the_sdk_client(self):
        stub = StubOpenAI()
        client = make_client(stub)
        await client.aclose()
        assert stub.closed

    async def test_context_manager_closes_the_client(self):
        stub = StubOpenAI()
        async with make_client(stub) as client:
            await client.acomplete([ChatMessage.user("hi")])
        assert stub.closed

    def test_registry_builds_an_openai_client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = build_llm_client(Settings(llm=settings()))
        assert isinstance(client, OpenAILLMClient)
        assert client.provider_name == "openai"


class TestProviderParity:
    """The two providers must present identical behaviour to the framework."""

    async def test_both_expose_the_same_response_shape(self):
        from llm_eval_guardrails.config import AWSSettings
        from llm_eval_guardrails.llm.bedrock import BedrockLLMClient

        class FakeBoto:
            @staticmethod
            def converse(**_request):
                return {
                    "output": {"message": {"content": [{"text": "hello"}]}},
                    "usage": {"inputTokens": 11, "outputTokens": 4},
                    "stopReason": "end_turn",
                }

        bedrock = BedrockLLMClient(
            LLMSettings(model_id="m", max_attempts=1),
            AWSSettings(),
            client=cast(Any, FakeBoto()),
        )
        openai_client = make_client(StubOpenAI())

        bedrock_response = await bedrock.acomplete([ChatMessage.user("hi")])
        openai_response = await openai_client.acomplete([ChatMessage.user("hi")])

        assert bedrock_response.text == openai_response.text
        assert bedrock_response.usage == openai_response.usage
        assert bedrock_response.latency_ms > 0
        assert openai_response.latency_ms > 0
