"""OpenAI evaluator-LLM client, used for local development and CI parity.

The official ``openai`` package is an optional dependency (``pip install
llm-eval-guardrails[openai]``). It is imported lazily so that a Bedrock-only
deployment neither installs nor loads it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..config import LLMSettings
from ..exceptions import ConfigurationError, LLMError, LLMRateLimitError, LLMTransientError
from ..logging_config import get_logger
from .base import ChatMessage, LLMClient, LLMResponse, TokenUsage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import AsyncOpenAI

__all__ = ["OpenAILLMClient"]

_log = get_logger(__name__)


class OpenAILLMClient(LLMClient):
    """Evaluator LLM backed by OpenAI's Chat Completions API.

    Retries and rate limiting are handled by the framework, so the SDK's own
    retry loop is disabled (``max_retries=0``) to avoid compounding delays.
    """

    provider_name = "openai"

    def __init__(self, settings: LLMSettings, *, client: AsyncOpenAI | None = None) -> None:
        """Initialise the OpenAI client.

        Args:
            settings: Evaluator LLM settings (model, sampling, retries, key).
            client: Pre-built ``AsyncOpenAI`` instance, primarily for tests.

        Raises:
            ConfigurationError: If the ``openai`` extra is not installed.
        """
        super().__init__(settings)
        self._client = client if client is not None else self._build_client(settings)

    @staticmethod
    def _build_client(settings: LLMSettings) -> AsyncOpenAI:
        """Construct an ``AsyncOpenAI`` client from settings.

        The API key falls back to the SDK's own ``OPENAI_API_KEY`` lookup when
        it is not set in framework configuration.

        Args:
            settings: Evaluator LLM settings.

        Returns:
            A configured async client.

        Raises:
            ConfigurationError: If the optional dependency is missing or no
                credentials could be resolved.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            msg = (
                "the OpenAI provider requires the 'openai' extra: "
                "pip install llm-eval-guardrails[openai]"
            )
            raise ConfigurationError(msg) from exc

        try:
            return AsyncOpenAI(
                api_key=(
                    settings.api_key.get_secret_value() if settings.api_key is not None else None
                ),
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
                max_retries=0,
            )
        except Exception as exc:
            msg = f"failed to create OpenAI client: {exc}"
            raise ConfigurationError(msg) from exc

    async def _invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
        model_id: str,
    ) -> LLMResponse:
        """Issue one Chat Completions request.

        Args:
            messages: The conversation turns.
            system: System instructions, prepended as a ``system`` role message.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.
            model_id: The OpenAI model name.

        Returns:
            The parsed completion.

        Raises:
            LLMRateLimitError: When the API reports rate limiting.
            LLMTransientError: On timeouts, connection errors and 5xx responses.
            LLMError: On any other API failure.
        """
        import openai

        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)

        request: dict[str, Any] = {
            "model": model_id,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._settings.top_p is not None:
            request["top_p"] = self._settings.top_p

        started = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**request)
        except openai.RateLimitError as exc:
            msg = f"openai rate limit: {exc}"
            raise LLMRateLimitError(msg, retry_after=_retry_after_seconds(exc)) from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            msg = f"openai transport failure: {exc}"
            raise LLMTransientError(msg) from exc
        except openai.APIStatusError as exc:
            msg = f"openai request failed [{exc.status_code}]: {exc}"
            if exc.status_code >= 500 or exc.status_code == 429:
                raise LLMTransientError(msg) from exc
            raise LLMError(msg) from exc
        except openai.OpenAIError as exc:
            msg = f"openai request failed: {exc}"
            raise LLMError(msg) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        return self._parse_completion(completion, model_id=model_id, latency_ms=latency_ms)

    @staticmethod
    def _parse_completion(completion: Any, *, model_id: str, latency_ms: float) -> LLMResponse:
        """Convert an OpenAI completion object into an :class:`LLMResponse`.

        Args:
            completion: The SDK's completion object.
            model_id: The model that was invoked.
            latency_ms: Measured provider latency.

        Returns:
            The normalised response.

        Raises:
            LLMTransientError: If the completion contains no usable text.
        """
        choices = getattr(completion, "choices", None) or []
        text = ""
        stop_reason: str | None = None
        if choices:
            text = getattr(choices[0].message, "content", None) or ""
            stop_reason = getattr(choices[0], "finish_reason", None)
        if not text.strip():
            msg = f"openai returned an empty completion (finish_reason={stop_reason})"
            raise LLMTransientError(msg)

        usage = getattr(completion, "usage", None)
        token_usage = TokenUsage(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
        return LLMResponse(
            text=text,
            model_id=getattr(completion, "model", model_id),
            usage=token_usage,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            raw={"id": getattr(completion, "id", None)},
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        close = getattr(self._client, "close", None)
        if callable(close):
            await close()


def _retry_after_seconds(exc: Any) -> float | None:
    """Extract a ``Retry-After`` hint from an OpenAI error, if present.

    Args:
        exc: The raised OpenAI exception.

    Returns:
        The hint in seconds, or ``None`` when absent or unparseable.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
