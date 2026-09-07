"""AWS Bedrock evaluator-LLM client built on the Converse API.

The Converse API is used in preference to ``invoke_model`` because it presents
one message schema across every Bedrock foundation model, so switching the
judge from Claude to another family requires only a configuration change.

boto3 is synchronous, so calls are dispatched to the default thread pool via
:func:`asyncio.to_thread`. Concurrency is bounded upstream by
:class:`~llm_eval_guardrails.llm.base.LLMClient`'s semaphore, which is sized to
match the botocore connection pool.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any, Final

from ..config import AWSSettings, LLMSettings
from ..exceptions import LLMError, LLMRateLimitError, LLMTransientError
from ..logging_config import get_logger
from .base import ChatMessage, LLMClient, LLMResponse, TokenUsage

#: botocore synthesises the ``converse`` operation at runtime from service
#: JSON, so no static type describes it. The alias keeps that ``Any`` explicit.
BedrockRuntimeClient = Any

__all__ = ["BedrockLLMClient"]

_log = get_logger(__name__)

#: Bedrock/botocore error codes that warrant a retry.
_RETRYABLE_CODES: Final[frozenset[str]] = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ServiceQuotaExceededException",
        "InternalServerException",
        "ModelTimeoutException",
        "ModelNotReadyException",
        "RequestTimeout",
        "RequestTimeoutException",
        "ConnectionError",
        "SlowDown",
        "503",
        "500",
    }
)

#: Error codes that specifically indicate rate limiting.
_RATE_LIMIT_CODES: Final[frozenset[str]] = frozenset(
    {"ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"}
)


class BedrockLLMClient(LLMClient):
    """Evaluator LLM backed by ``bedrock-runtime``'s ``Converse`` operation.

    Authentication uses the standard AWS credential chain (environment, shared
    config, IRSA/EKS web identity, or the ECS/EC2 instance role), so no API key
    is stored by the framework.

    Example:
        >>> client = BedrockLLMClient(llm_settings, aws_settings)  # doctest: +SKIP
        >>> response = await client.acomplete([ChatMessage.user("Hi")])  # doctest: +SKIP
    """

    provider_name = "bedrock"

    def __init__(
        self,
        settings: LLMSettings,
        aws_settings: AWSSettings,
        *,
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialise the Bedrock client.

        Args:
            settings: Evaluator LLM settings (model, sampling, retries).
            aws_settings: Region, profile and connection-pool settings.
            client: Pre-built ``bedrock-runtime`` client. Supplying one keeps
                unit tests free of AWS credentials and network access.
        """
        super().__init__(settings)
        self._aws_settings = aws_settings
        self._client = client if client is not None else self._build_client(settings, aws_settings)

    @staticmethod
    def _build_client(settings: LLMSettings, aws_settings: AWSSettings) -> BedrockRuntimeClient:
        """Construct a configured ``bedrock-runtime`` client.

        botocore's own retry handling is disabled (``max_attempts=0``) because
        this framework applies its own tenacity policy; layering the two would
        multiply the effective attempt count and blow past the configured
        latency budget.

        Args:
            settings: Evaluator LLM settings supplying the request timeout.
            aws_settings: Region, profile and pool configuration.

        Returns:
            A ready-to-use boto3 client.

        Raises:
            LLMError: If the AWS SDK could not create the client.
        """
        import boto3
        from botocore.config import Config

        botocore_config = Config(
            region_name=aws_settings.region,
            retries={"max_attempts": 0, "mode": "standard"},
            connect_timeout=min(10.0, settings.timeout_seconds),
            read_timeout=settings.timeout_seconds,
            max_pool_connections=max(aws_settings.max_pool_connections, settings.max_in_flight),
            user_agent_extra="llm-eval-guardrails/0.1.0",
        )
        try:
            session = boto3.Session(
                profile_name=aws_settings.profile, region_name=aws_settings.region
            )
            client: BedrockRuntimeClient = session.client(
                "bedrock-runtime",
                config=botocore_config,
                endpoint_url=settings.base_url,
            )
        except Exception as exc:
            msg = f"failed to create bedrock-runtime client: {exc}"
            raise LLMError(msg) from exc
        return client

    async def _invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
        model_id: str,
    ) -> LLMResponse:
        """Issue one ``Converse`` request.

        Args:
            messages: The conversation turns.
            system: System instructions, or ``None``.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.
            model_id: Bedrock model or inference-profile identifier.

        Returns:
            The parsed completion.

        Raises:
            LLMTransientError: On throttling, timeouts or 5xx responses.
            LLMError: On any other Bedrock failure.
        """
        request: dict[str, Any] = {
            "modelId": model_id,
            "messages": [{"role": m.role, "content": [{"text": m.content}]} for m in messages],
            "inferenceConfig": self._inference_config(temperature, max_tokens),
        }
        if system:
            request["system"] = [{"text": system}]

        started = time.perf_counter()
        payload = await asyncio.to_thread(self._converse, request)
        latency_ms = (time.perf_counter() - started) * 1000
        return self._parse_response(payload, model_id=model_id, latency_ms=latency_ms)

    def _inference_config(self, temperature: float, max_tokens: int) -> dict[str, Any]:
        """Build the Converse ``inferenceConfig`` block.

        Args:
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.

        Returns:
            The inference configuration, omitting unset optional fields.
        """
        config: dict[str, Any] = {"maxTokens": max_tokens, "temperature": temperature}
        if self._settings.top_p is not None:
            config["topP"] = self._settings.top_p
        return config

    def _converse(self, request: dict[str, Any]) -> dict[str, Any]:
        """Call ``Converse`` synchronously and normalise provider errors.

        Args:
            request: The fully-formed Converse request.

        Returns:
            The raw Bedrock response payload.

        Raises:
            LLMRateLimitError: When Bedrock reports throttling.
            LLMTransientError: On other retryable failures.
            LLMError: On permanent failures such as validation or access errors.
        """
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            response: dict[str, Any] = self._client.converse(**request)
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = str(error.get("Code", "Unknown"))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
            message = "bedrock converse failed [{code}]: {msg}".format(
                code=code, msg=error.get("Message", str(exc))
            )
            if code in _RATE_LIMIT_CODES:
                raise LLMRateLimitError(message) from exc
            if code in _RETRYABLE_CODES or status >= 500 or status == 429:
                raise LLMTransientError(message) from exc
            raise LLMError(message) from exc
        except BotoCoreError as exc:
            # Connection resets, DNS failures and read timeouts all land here.
            msg = f"bedrock transport failure: {exc}"
            raise LLMTransientError(msg) from exc
        return response

    @staticmethod
    def _parse_response(
        payload: dict[str, Any], *, model_id: str, latency_ms: float
    ) -> LLMResponse:
        """Convert a Converse payload into an :class:`LLMResponse`.

        Args:
            payload: The raw Bedrock response.
            model_id: The model that was invoked.
            latency_ms: Measured provider latency.

        Returns:
            The normalised response.

        Raises:
            LLMTransientError: If the response contains no text block, which in
                practice means the model returned an empty or tool-only turn
                and is worth one more attempt.
        """
        content = payload.get("output", {}).get("message", {}).get("content", [])
        text = "".join(
            block["text"] for block in content if isinstance(block, dict) and "text" in block
        )
        if not text.strip():
            msg = "bedrock returned an empty completion (stopReason={reason})".format(
                reason=payload.get("stopReason")
            )
            raise LLMTransientError(msg)

        usage_block = payload.get("usage", {})
        return LLMResponse(
            text=text,
            model_id=model_id,
            usage=TokenUsage(
                input_tokens=int(usage_block.get("inputTokens", 0) or 0),
                output_tokens=int(usage_block.get("outputTokens", 0) or 0),
            ),
            latency_ms=latency_ms,
            stop_reason=payload.get("stopReason"),
            raw={k: v for k, v in payload.items() if k != "ResponseMetadata"},
        )

    async def aclose(self) -> None:
        """Close the underlying botocore client and release its sockets."""
        close = getattr(self._client, "close", None)
        if callable(close):
            await asyncio.to_thread(close)
