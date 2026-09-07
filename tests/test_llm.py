"""Tests for the LLM abstraction, retries, rate limiting and providers."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from pydantic import BaseModel

from llm_eval_guardrails.config import AWSSettings, LLMProvider, LLMSettings, Settings
from llm_eval_guardrails.exceptions import (
    ConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMTransientError,
)
from llm_eval_guardrails.llm.base import (
    ChatMessage,
    LLMClient,
    TokenUsage,
    extract_json_object,
)
from llm_eval_guardrails.llm.bedrock import BedrockLLMClient
from llm_eval_guardrails.llm.rate_limit import AsyncRateLimiter
from llm_eval_guardrails.llm.registry import build_llm_client, register_provider

from .conftest import FakeLLMClient


class Verdict(BaseModel):
    score: float
    reasoning: str | None = None


class TestExtractJsonObject:
    def test_parses_bare_json(self):
        assert extract_json_object('{"score": 0.5}') == {"score": 0.5}

    def test_parses_fenced_json(self):
        text = 'Here you go:\n```json\n{"score": 0.5}\n```\nHope that helps!'
        assert extract_json_object(text) == {"score": 0.5}

    def test_parses_unlabelled_fence(self):
        assert extract_json_object('```\n{"score": 1}\n```') == {"score": 1}

    def test_parses_json_embedded_in_prose(self):
        assert extract_json_object('Sure. {"score": 0.25} Done.') == {"score": 0.25}

    def test_handles_braces_inside_string_values(self):
        text = 'prose {"reasoning": "uses {braces} inside", "score": 1} tail'
        assert extract_json_object(text)["reasoning"] == "uses {braces} inside"

    def test_handles_escaped_quotes(self):
        text = r'{"reasoning": "he said \"hi\"", "score": 1}'
        assert extract_json_object(text)["reasoning"] == 'he said "hi"'

    def test_handles_nested_objects(self):
        payload = {"a": {"b": {"c": 1}}, "score": 1}
        assert extract_json_object(f"text {json.dumps(payload)} more") == payload

    def test_rejects_a_bare_json_array(self):
        with pytest.raises(LLMResponseFormatError):
            extract_json_object("[1, 2, 3]")

    def test_rejects_non_json(self):
        with pytest.raises(LLMResponseFormatError, match="no JSON object"):
            extract_json_object("I cannot comply with that request.")

    def test_rejects_empty_string(self):
        with pytest.raises(LLMResponseFormatError):
            extract_json_object("")

    def test_rejects_unterminated_object(self):
        with pytest.raises(LLMResponseFormatError):
            extract_json_object('{"score": 0.5')


class TestTokenUsage:
    def test_total_is_the_sum(self):
        assert TokenUsage(input_tokens=10, output_tokens=5).total_tokens == 15

    def test_addition_is_elementwise(self):
        combined = TokenUsage(1, 2) + TokenUsage(3, 4)
        assert combined == TokenUsage(4, 6)


class TestChatMessage:
    def test_rejects_an_unknown_role(self):
        with pytest.raises(ValueError, match="role must be"):
            ChatMessage(role="system", content="x")

    def test_constructors_set_the_role(self):
        assert ChatMessage.user("x").role == "user"
        assert ChatMessage.assistant("x").role == "assistant"


class TestLLMClientBehaviour:
    async def test_records_usage_and_call_count(self, llm_settings):
        llm = FakeLLMClient(['{"score": 1}', '{"score": 1}'], llm_settings)
        await llm.acomplete([ChatMessage.user("a")])
        await llm.acomplete([ChatMessage.user("b")])
        assert llm.call_count == 2
        assert llm.usage.total_tokens == 30

    async def test_rejects_empty_message_list(self, fake_llm):
        with pytest.raises(ValueError, match="must not be empty"):
            await fake_llm.acomplete([])

    async def test_counts_failures(self, llm_settings):
        llm = FakeLLMClient([LLMError("permanent")], llm_settings)
        with pytest.raises(LLMError):
            await llm.acomplete([ChatMessage.user("a")])
        assert llm.failure_count == 1
        assert llm.call_count == 0

    async def test_retries_transient_errors(self):
        settings = LLMSettings(
            max_attempts=3, initial_backoff_seconds=0.001, max_backoff_seconds=0.002
        )
        llm = FakeLLMClient(
            [LLMTransientError("throttled"), LLMTransientError("throttled"), '{"score": 1}'],
            settings,
        )
        response = await llm.acomplete([ChatMessage.user("a")])
        assert response.text == '{"score": 1}'
        assert llm.call_count == 1

    async def test_does_not_retry_permanent_errors(self):
        settings = LLMSettings(max_attempts=5, initial_backoff_seconds=0.001)
        llm = FakeLLMClient([LLMError("validation failed"), '{"score": 1}'], settings)
        with pytest.raises(LLMError):
            await llm.acomplete([ChatMessage.user("a")])
        assert len(llm.prompts) == 1

    async def test_gives_up_after_max_attempts(self):
        settings = LLMSettings(
            max_attempts=2, initial_backoff_seconds=0.001, max_backoff_seconds=0.002
        )
        llm = FakeLLMClient([LLMTransientError("throttled")], settings)
        with pytest.raises(LLMTransientError):
            await llm.acomplete([ChatMessage.user("a")])
        assert len(llm.prompts) == 2

    async def test_latency_is_populated(self, fake_llm):
        response = await fake_llm.acomplete([ChatMessage.user("a")])
        assert response.latency_ms > 0

    async def test_estimated_cost_is_none_without_pricing(self, fake_llm):
        await fake_llm.acomplete([ChatMessage.user("a")])
        assert fake_llm.estimated_cost_usd() is None

    async def test_estimated_cost_uses_configured_pricing(self):
        settings = LLMSettings(max_attempts=1, input_cost_per_1k=1.0, output_cost_per_1k=2.0)
        llm = FakeLLMClient(['{"score": 1}'], settings)
        await llm.acomplete([ChatMessage.user("a")])
        # 10 input + 5 output tokens at 1.0/2.0 per 1K.
        assert llm.estimated_cost_usd() == pytest.approx(0.01 + 0.01)

    async def test_concurrency_is_capped_by_max_in_flight(self):
        settings = LLMSettings(max_attempts=1, max_in_flight=2, requests_per_minute=None)

        class CountingClient(FakeLLMClient):
            def __init__(self):
                super().__init__(['{"score": 1}'], settings)
                self.in_flight = 0
                self.peak = 0

            async def _invoke(self, messages, **kwargs):
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
                await asyncio.sleep(0.01)
                self.in_flight -= 1
                return await super()._invoke(messages, **kwargs)

        client = CountingClient()
        await asyncio.gather(*(client.acomplete([ChatMessage.user("x")]) for _ in range(8)))
        assert client.peak <= 2


class TestCompleteJson:
    async def test_returns_a_validated_model(self, llm_settings):
        llm = FakeLLMClient(['{"score": 0.75, "reasoning": "ok"}'], llm_settings)
        parsed, response = await llm.acomplete_json([ChatMessage.user("a")], schema=Verdict)
        assert parsed.score == 0.75
        assert response.text.startswith("{")

    async def test_repairs_a_malformed_first_response(self, llm_settings):
        llm = FakeLLMClient(["sorry, no JSON", '{"score": 0.5}'], llm_settings)
        parsed, _ = await llm.acomplete_json([ChatMessage.user("a")], schema=Verdict)
        assert parsed.score == 0.5
        assert len(llm.prompts) == 2

    async def test_repair_prompt_includes_the_schema(self, llm_settings):
        llm = FakeLLMClient(["nope", '{"score": 0.5}'], llm_settings)
        await llm.acomplete_json([ChatMessage.user("a")], schema=Verdict)
        assert "JSON Schema" in llm.prompts[1]

    async def test_gives_up_after_repair_budget(self, llm_settings):
        llm = FakeLLMClient(["nope", "still nope", "nope again"], llm_settings)
        with pytest.raises(LLMResponseFormatError, match="valid Verdict JSON"):
            await llm.acomplete_json([ChatMessage.user("a")], schema=Verdict, repair_attempts=1)
        assert len(llm.prompts) == 2

    async def test_schema_violations_trigger_repair(self, llm_settings):
        llm = FakeLLMClient(['{"reasoning": "no score field"}', '{"score": 1}'], llm_settings)
        parsed, _ = await llm.acomplete_json([ChatMessage.user("a")], schema=Verdict)
        assert parsed.score == 1


class TestSyncFacade:
    def test_complete_works_outside_a_loop(self, llm_settings):
        llm = FakeLLMClient(['{"score": 1}'], llm_settings)
        assert llm.complete([ChatMessage.user("a")]).text == '{"score": 1}'

    async def test_complete_refuses_inside_a_loop(self, fake_llm):
        with pytest.raises(RuntimeError, match="running event loop"):
            fake_llm.complete([ChatMessage.user("a")])

    def test_complete_json_works_outside_a_loop(self, llm_settings):
        llm = FakeLLMClient(['{"score": 0.4}'], llm_settings)
        parsed, _ = llm.complete_json([ChatMessage.user("a")], schema=Verdict)
        assert parsed.score == 0.4


class TestAsyncRateLimiter:
    async def test_disabled_limiter_never_waits(self):
        limiter = AsyncRateLimiter(None)
        assert not limiter.enabled
        assert await limiter.acquire() == 0.0

    async def test_zero_rate_disables_the_limiter(self):
        assert not AsyncRateLimiter(0).enabled

    async def test_burst_is_served_without_waiting(self):
        limiter = AsyncRateLimiter(6000, burst=5)
        started = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        assert time.monotonic() - started < 0.1

    async def test_exhausted_bucket_forces_a_wait(self):
        limiter = AsyncRateLimiter(600, burst=1)  # 10 tokens/second
        await limiter.acquire()
        started = time.monotonic()
        await limiter.acquire()
        assert time.monotonic() - started >= 0.05

    async def test_requesting_more_than_capacity_is_rejected(self):
        limiter = AsyncRateLimiter(60, burst=1)
        with pytest.raises(ValueError, match="exceeds bucket capacity"):
            await limiter.acquire(tokens=10)

    async def test_context_manager_acquires(self):
        limiter = AsyncRateLimiter(6000, burst=2)
        async with limiter:
            pass


class TestRetryAfterHint:
    async def test_provider_hint_is_honoured_when_larger(self):
        settings = LLMSettings(
            max_attempts=2, initial_backoff_seconds=0.001, max_backoff_seconds=0.002
        )
        llm = FakeLLMClient(
            [LLMRateLimitError("slow down", retry_after=0.05), '{"score": 1}'], settings
        )
        started = time.monotonic()
        await llm.acomplete([ChatMessage.user("a")])
        assert time.monotonic() - started >= 0.05


class TestBedrockClient:
    def _client(self, converse_result=None, error=None):
        class FakeBotoClient:
            def __init__(self):
                self.requests = []

            def converse(self, **request):
                self.requests.append(request)
                if error is not None:
                    raise error
                return converse_result

        return FakeBotoClient()

    def _settings(self):
        return LLMSettings(model_id="test-model", max_attempts=1, requests_per_minute=None)

    async def test_parses_a_converse_response(self):
        boto = self._client(
            {
                "output": {"message": {"content": [{"text": "hello"}]}},
                "usage": {"inputTokens": 7, "outputTokens": 3},
                "stopReason": "end_turn",
            }
        )
        client = BedrockLLMClient(self._settings(), AWSSettings(), client=boto)
        response = await client.acomplete([ChatMessage.user("hi")])
        assert response.text == "hello"
        assert response.usage.input_tokens == 7
        assert response.stop_reason == "end_turn"

    async def test_builds_a_well_formed_request(self):
        boto = self._client({"output": {"message": {"content": [{"text": "x"}]}}, "usage": {}})
        client = BedrockLLMClient(self._settings(), AWSSettings(), client=boto)
        await client.acomplete([ChatMessage.user("hi")], system="be terse")
        request = boto.requests[0]
        assert request["modelId"] == "test-model"
        assert request["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
        assert request["system"] == [{"text": "be terse"}]
        assert request["inferenceConfig"]["maxTokens"] > 0

    async def test_omits_the_system_block_when_absent(self):
        boto = self._client({"output": {"message": {"content": [{"text": "x"}]}}, "usage": {}})
        client = BedrockLLMClient(self._settings(), AWSSettings(), client=boto)
        await client.acomplete([ChatMessage.user("hi")])
        assert "system" not in boto.requests[0]

    async def test_concatenates_multiple_text_blocks(self):
        boto = self._client(
            {
                "output": {"message": {"content": [{"text": "a"}, {"text": "b"}]}},
                "usage": {},
            }
        )
        client = BedrockLLMClient(self._settings(), AWSSettings(), client=boto)
        assert (await client.acomplete([ChatMessage.user("hi")])).text == "ab"

    async def test_empty_completion_is_transient(self):
        boto = self._client(
            {"output": {"message": {"content": []}}, "usage": {}, "stopReason": "max_tokens"}
        )
        client = BedrockLLMClient(self._settings(), AWSSettings(), client=boto)
        with pytest.raises(LLMTransientError, match="empty completion"):
            await client.acomplete([ChatMessage.user("hi")])

    async def test_throttling_maps_to_rate_limit_error(self):
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "Converse"
        )
        client = BedrockLLMClient(self._settings(), AWSSettings(), client=self._client(error=error))
        with pytest.raises(LLMRateLimitError):
            await client.acomplete([ChatMessage.user("hi")])

    async def test_validation_error_is_permanent(self):
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "bad input"}}, "Converse"
        )
        client = BedrockLLMClient(self._settings(), AWSSettings(), client=self._client(error=error))
        with pytest.raises(LLMError) as excinfo:
            await client.acomplete([ChatMessage.user("hi")])
        assert not isinstance(excinfo.value, LLMTransientError)

    async def test_server_error_is_transient(self):
        from botocore.exceptions import ClientError

        response: Any = {
            "Error": {"Code": "SomethingElse", "Message": "boom"},
            "ResponseMetadata": {"HTTPStatusCode": 503},
        }
        error = ClientError(response, "Converse")
        client = BedrockLLMClient(self._settings(), AWSSettings(), client=self._client(error=error))
        with pytest.raises(LLMTransientError):
            await client.acomplete([ChatMessage.user("hi")])

    async def test_transport_failure_is_transient(self):
        from botocore.exceptions import EndpointConnectionError

        error = EndpointConnectionError(endpoint_url="https://bedrock.example")
        client = BedrockLLMClient(self._settings(), AWSSettings(), client=self._client(error=error))
        with pytest.raises(LLMTransientError):
            await client.acomplete([ChatMessage.user("hi")])


class TestRegistry:
    def test_builds_a_bedrock_client_by_default(self, monkeypatch):
        built = {}

        def fake_init(self, settings, aws_settings, *, client=None):
            built["called"] = True
            # Skip the real __init__ (which would build a boto3 client) while
            # still initialising the shared LLMClient machinery.
            LLMClient.__init__(self, settings)

        monkeypatch.setattr(BedrockLLMClient, "__init__", fake_init)
        client = build_llm_client(Settings())
        assert isinstance(client, BedrockLLMClient)
        assert built["called"]

    def test_openai_without_the_extra_raises_configuration_error(self):
        settings = Settings(llm=LLMSettings(provider=LLMProvider.OPENAI))
        try:
            import openai  # noqa: F401
        except ImportError:
            with pytest.raises(ConfigurationError, match="openai"):
                build_llm_client(settings)
        else:
            pytest.skip("the openai extra is installed in this environment")

    def test_custom_provider_can_be_registered(self, llm_settings):
        sentinel = FakeLLMClient(settings=llm_settings)
        register_provider("custom", lambda _settings: sentinel)
        settings = Settings()
        settings.llm.provider = "custom"  # type: ignore[assignment]
        assert build_llm_client(settings) is sentinel

    def test_registering_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            register_provider("  ", lambda s: None)  # type: ignore[arg-type,return-value]
