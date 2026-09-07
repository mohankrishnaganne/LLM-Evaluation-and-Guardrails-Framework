"""Provider-agnostic abstraction for evaluator LLMs (LLM-as-a-judge).

Concrete providers implement a single async primitive, :meth:`LLMClient._invoke`,
and inherit retries, rate limiting, latency logging, usage accounting, a
synchronous facade, and strict JSON parsing from :class:`LLMClient`.

The contract is deliberately narrow: the framework only ever needs
"send messages, get text back", so the abstraction does not leak
provider-specific concepts such as Bedrock inference profiles or OpenAI
response formats.
"""

from __future__ import annotations

import abc
import asyncio
import json
import re
import time
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import LLMSettings
from ..exceptions import LLMResponseFormatError
from ..logging_config import get_logger
from .rate_limit import AsyncRateLimiter, retry_policy

__all__ = [
    "ChatMessage",
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
    "extract_json_object",
]

_log = get_logger(__name__)

_ModelT = TypeVar("_ModelT", bound=BaseModel)

#: Matches a ```json ... ``` fenced block, tolerating a missing language tag.
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(?P<body>.*?)```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single conversational turn sent to the evaluator LLM.

    Attributes:
        role: Either ``"user"`` or ``"assistant"``. System instructions are
            passed separately because providers model them differently.
        content: The message text.
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        """Validate the role.

        Raises:
            ValueError: If the role is not ``"user"`` or ``"assistant"``.
        """
        if self.role not in {"user", "assistant"}:
            msg = f"role must be 'user' or 'assistant', got {self.role!r}"
            raise ValueError(msg)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        """Build a user turn.

        Args:
            content: The message text.

        Returns:
            The constructed message.
        """
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        """Build an assistant turn.

        Args:
            content: The message text.

        Returns:
            The constructed message.
        """
        return cls(role="assistant", content=content)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for one or more LLM calls.

    Attributes:
        input_tokens: Tokens consumed by the prompt.
        output_tokens: Tokens produced in the completion.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens across both directions.

        Returns:
            The sum of input and output tokens.
        """
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Combine two usage records.

        Args:
            other: The usage to add.

        Returns:
            A new record holding the element-wise sum.
        """
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A completion returned by an evaluator LLM.

    Attributes:
        text: The generated text.
        model_id: The model that produced the completion.
        usage: Token accounting, zero-filled when the provider omits it.
        latency_ms: Wall-clock latency of the provider call.
        stop_reason: Provider-reported termination reason, when available.
        raw: The unmodified provider payload, retained for audit.
    """

    text: str
    model_id: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model output.

    Judge models reliably emit JSON but frequently wrap it in prose or a
    Markdown fence. This helper tries, in order: a direct parse, the contents
    of a fenced block, and finally the outermost brace-balanced span (ignoring
    braces inside string literals).

    Args:
        text: Raw text returned by the model.

    Returns:
        The decoded JSON object.

    Raises:
        LLMResponseFormatError: If no JSON object can be recovered.
    """
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    fence = _FENCE_RE.search(text)
    if fence is not None:
        candidates.append(fence.group("body").strip())

    span = _outermost_object(text)
    if span is not None:
        candidates.append(span)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = text[:300].replace("\n", " ")
    msg = f"no JSON object found in model output: {preview!r}"
    raise LLMResponseFormatError(msg)


def _outermost_object(text: str) -> str | None:
    """Return the outermost brace-balanced JSON object substring.

    String literals and escape sequences are tracked so that braces appearing
    inside quoted values do not unbalance the scan.

    Args:
        text: Text that may embed a JSON object.

    Returns:
        The balanced substring, or ``None`` when no complete object is present.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


class LLMClient(abc.ABC):
    """Base class for evaluator-LLM providers.

    Subclasses implement :meth:`_invoke`; everything else -- retries with
    exponential backoff and jitter, client-side rate limiting, concurrency
    capping, structured latency logging, cumulative usage accounting, the
    synchronous facade and schema-validated JSON parsing -- is provided here.

    The client is safe to share across concurrent tasks. Instances own an
    :class:`asyncio.Semaphore`, so construct them inside the event loop that
    will use them (or at least do not move one between running loops).
    """

    #: Human-readable provider name, surfaced in logs and reports.
    provider_name: str = "base"

    def __init__(self, settings: LLMSettings) -> None:
        """Initialise shared retry, rate-limiting and accounting machinery.

        Args:
            settings: Provider configuration.
        """
        self._settings = settings
        self._limiter = AsyncRateLimiter(settings.requests_per_minute)
        self._semaphore = asyncio.Semaphore(settings.max_in_flight)
        self._usage = TokenUsage()
        self._call_count = 0
        self._failure_count = 0
        self._usage_lock = asyncio.Lock()
        self._log = _log.bind(provider=self.provider_name, model_id=settings.model_id)

    # ------------------------------------------------------------------ #
    # Provider contract
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def _invoke(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
        model_id: str,
    ) -> LLMResponse:
        """Issue a single, un-retried completion request.

        Implementations must translate provider errors into the framework's
        exception hierarchy: retryable conditions (throttling, 5xx, timeouts)
        as :class:`~llm_eval_guardrails.exceptions.LLMTransientError` and
        permanent ones as :class:`~llm_eval_guardrails.exceptions.LLMError`.

        Args:
            messages: The conversation turns.
            system: System instructions, or ``None``.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.
            model_id: The model to invoke.

        Returns:
            The provider's completion.
        """

    async def aclose(self) -> None:
        """Release provider resources.

        The default implementation does nothing; providers holding sockets or
        SDK clients should override it.
        """
        return

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def model_id(self) -> str:
        """The configured default model identifier.

        Returns:
            The model id from settings.
        """
        return self._settings.model_id

    @property
    def usage(self) -> TokenUsage:
        """Cumulative token usage across every call made by this client.

        Returns:
            The aggregated usage record.
        """
        return self._usage

    @property
    def call_count(self) -> int:
        """Number of successful completions issued by this client.

        Returns:
            The successful call count.
        """
        return self._call_count

    @property
    def failure_count(self) -> int:
        """Number of calls that exhausted retries and failed.

        Returns:
            The failed call count.
        """
        return self._failure_count

    def estimated_cost_usd(self) -> float | None:
        """Estimate spend from cumulative usage and configured pricing.

        Returns:
            The estimated cost in USD, or ``None`` when pricing is unset.
        """
        input_rate = self._settings.input_cost_per_1k
        output_rate = self._settings.output_cost_per_1k
        if input_rate is None and output_rate is None:
            return None
        cost = (self._usage.input_tokens / 1000.0) * (input_rate or 0.0)
        cost += (self._usage.output_tokens / 1000.0) * (output_rate or 0.0)
        return cost

    async def acomplete(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_id: str | None = None,
    ) -> LLMResponse:
        """Generate a completion, with rate limiting and retries applied.

        Args:
            messages: The conversation turns; must be non-empty.
            system: System instructions, or ``None``.
            temperature: Override the configured temperature.
            max_tokens: Override the configured completion budget.
            model_id: Override the configured model.

        Returns:
            The provider's completion.

        Raises:
            ValueError: If ``messages`` is empty.
            LLMError: If the call fails after exhausting all attempts.
        """
        if not messages:
            msg = "messages must not be empty"
            raise ValueError(msg)

        effective_model = model_id or self._settings.model_id
        effective_temperature = self._settings.temperature if temperature is None else temperature
        effective_max_tokens = max_tokens or self._settings.max_tokens

        async def _guarded_invoke() -> LLMResponse:
            """Run one attempt under the concurrency cap and rate limiter.

            Returns:
                The provider's completion for this attempt.
            """
            async with self._semaphore:
                await self._limiter.acquire()
                return await self._invoke(
                    messages,
                    system=system,
                    temperature=effective_temperature,
                    max_tokens=effective_max_tokens,
                    model_id=effective_model,
                )

        started = time.perf_counter()
        try:
            response: LLMResponse = await retry_policy(self._settings)(_guarded_invoke)
        except Exception:
            self._failure_count += 1
            self._log.warning(
                "llm.call_failed",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                exc_info=True,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        async with self._usage_lock:
            self._usage = self._usage + response.usage
            self._call_count += 1

        self._log.debug(
            "llm.call_completed",
            latency_ms=round(elapsed_ms, 2),
            provider_latency_ms=round(response.latency_ms, 2),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )
        if response.latency_ms:
            return response
        return replace(response, latency_ms=elapsed_ms)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_id: str | None = None,
    ) -> LLMResponse:
        """Synchronous facade over :meth:`acomplete` for real-time spot checks.

        Args:
            messages: The conversation turns.
            system: System instructions, or ``None``.
            temperature: Override the configured temperature.
            max_tokens: Override the configured completion budget.
            model_id: Override the configured model.

        Returns:
            The provider's completion.

        Raises:
            RuntimeError: If called from inside a running event loop, where it
                would deadlock. Use :meth:`acomplete` there instead.
        """
        return _run_sync(
            self.acomplete(
                messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                model_id=model_id,
            )
        )

    async def acomplete_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: type[_ModelT],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_id: str | None = None,
        repair_attempts: int = 1,
    ) -> tuple[_ModelT, LLMResponse]:
        """Generate a completion and validate it against a Pydantic schema.

        When parsing or validation fails, the model is re-prompted with the
        offending output and the validation error, which recovers the common
        case of a judge emitting prose alongside its JSON. Repair prompts are
        bounded by ``repair_attempts`` so a persistently malformed judge fails
        fast instead of burning the rate-limit budget.

        Args:
            messages: The conversation turns.
            schema: The Pydantic model the response must satisfy.
            system: System instructions, or ``None``.
            temperature: Override the configured temperature.
            max_tokens: Override the configured completion budget.
            model_id: Override the configured model.
            repair_attempts: Additional re-prompts allowed after a parse or
                validation failure.

        Returns:
            A tuple of the validated model and the final raw response.

        Raises:
            LLMResponseFormatError: If valid JSON matching ``schema`` could not
                be obtained within the allotted attempts.
        """
        turns = list(messages)
        last_error: Exception | None = None

        for attempt_index in range(repair_attempts + 1):
            response = await self.acomplete(
                turns,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                model_id=model_id,
            )
            try:
                payload = extract_json_object(response.text)
                return schema.model_validate(payload), response
            except (LLMResponseFormatError, ValidationError) as exc:
                last_error = exc
                self._log.warning(
                    "llm.json_parse_failed",
                    attempt=attempt_index + 1,
                    schema=schema.__name__,
                    error=str(exc)[:500],
                )
                if attempt_index >= repair_attempts:
                    break
                turns = [
                    *turns,
                    ChatMessage.assistant(response.text),
                    ChatMessage.user(_repair_prompt(schema, exc)),
                ]

        msg = (
            f"judge did not return valid {schema.__name__} JSON after "
            f"{repair_attempts + 1} attempt(s): {last_error}"
        )
        raise LLMResponseFormatError(msg) from last_error

    def complete_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        schema: type[_ModelT],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_id: str | None = None,
        repair_attempts: int = 1,
    ) -> tuple[_ModelT, LLMResponse]:
        """Synchronous facade over :meth:`acomplete_json`.

        Args:
            messages: The conversation turns.
            schema: The Pydantic model the response must satisfy.
            system: System instructions, or ``None``.
            temperature: Override the configured temperature.
            max_tokens: Override the configured completion budget.
            model_id: Override the configured model.
            repair_attempts: Additional re-prompts allowed on parse failure.

        Returns:
            A tuple of the validated model and the final raw response.
        """
        return _run_sync(
            self.acomplete_json(
                messages,
                schema=schema,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                model_id=model_id,
                repair_attempts=repair_attempts,
            )
        )

    async def __aenter__(self) -> LLMClient:
        """Enter the async context manager.

        Returns:
            This client.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close provider resources on context exit.

        Args:
            exc_type: Exception type, if the block raised.
            exc: Exception instance, if the block raised.
            tb: Traceback, if the block raised.
        """
        await self.aclose()


def _repair_prompt(schema: type[BaseModel], error: Exception) -> str:
    """Build a re-prompt asking the judge to emit schema-valid JSON.

    Args:
        schema: The expected response model.
        error: The parse or validation failure encountered.

    Returns:
        The repair instruction to append as a user turn.
    """
    return (
        "Your previous response could not be parsed.\n"
        f"Error: {str(error)[:800]}\n\n"
        "Respond again with ONLY a single JSON object matching this JSON Schema. "
        "Emit no prose, no explanation and no Markdown code fence.\n\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )


_T = TypeVar("_T")


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine to completion from synchronous code.

    Args:
        coro: The coroutine to run.

    Returns:
        The coroutine's result.

    Raises:
        RuntimeError: If an event loop is already running in this thread, in
            which case awaiting the async API directly is required.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    coro.close()
    msg = (
        "synchronous API called from a running event loop; "
        "await the async equivalent (acomplete/acomplete_json/aevaluate) instead"
    )
    raise RuntimeError(msg)
