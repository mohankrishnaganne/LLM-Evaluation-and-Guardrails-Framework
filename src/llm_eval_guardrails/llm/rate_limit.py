"""Retry and client-side rate-limiting primitives for external LLM calls.

Two independent concerns are handled here:

* **Retries** -- :func:`retry_policy` builds a :class:`tenacity.AsyncRetrying`
  configured for exponential backoff with full jitter, bounded attempts and a
  provider-aware predicate that only retries transient failures.
* **Rate limiting** -- :class:`AsyncRateLimiter` implements a token bucket that
  smooths request bursts to stay inside a provider's requests-per-minute quota.

Keeping these out of the provider clients means Bedrock and OpenAI share
identical resilience behaviour.
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType

from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_base, wait_exponential_jitter

from ..config import LLMSettings
from ..exceptions import LLMTransientError
from ..logging_config import get_logger

__all__ = ["AsyncRateLimiter", "retry_policy"]

_log = get_logger(__name__)


def _log_retry(state: RetryCallState) -> None:
    """Emit a structured event before each retry sleep.

    Args:
        state: Tenacity's state for the attempt that just failed.
    """
    outcome = state.outcome
    error = outcome.exception() if outcome is not None and outcome.failed else None
    _log.warning(
        "llm.retry",
        attempt=state.attempt_number,
        sleep_seconds=round(state.idle_for, 3),
        error_type=type(error).__name__ if error is not None else None,
        error=str(error)[:300] if error is not None else None,
    )


def retry_policy(settings: LLMSettings) -> AsyncRetrying:
    """Build the shared retry policy for evaluator-LLM calls.

    Only :class:`~llm_eval_guardrails.exceptions.LLMTransientError` (and its
    subclass ``LLMRateLimitError``) is retried, so malformed requests and
    authorisation failures surface immediately instead of amplifying load
    against a provider that is already rejecting the call.

    Args:
        settings: Provider settings supplying attempt and backoff bounds.

    Returns:
        A configured :class:`tenacity.AsyncRetrying` instance. It re-raises the
        final exception rather than wrapping it in ``RetryError``, so callers
        see the provider's own error type.
    """
    return AsyncRetrying(
        retry=retry_if_exception_type(LLMTransientError),
        stop=stop_after_attempt(settings.max_attempts),
        wait=_wait_with_retry_after(
            wait_exponential_jitter(
                initial=settings.initial_backoff_seconds,
                max=settings.max_backoff_seconds,
                jitter=settings.initial_backoff_seconds,
            )
        ),
        before_sleep=_log_retry,
        reraise=True,
    )


class _wait_with_retry_after(wait_base):  # noqa: N801 - matches tenacity naming
    """Honour a provider-supplied ``Retry-After`` hint when one is present.

    Providers that return an explicit backoff hint (as ``retry_after`` on the
    raised :class:`~llm_eval_guardrails.exceptions.LLMTransientError`) know more
    about their own queue depth than an exponential schedule does. The hint is
    used when it exceeds the computed delay; otherwise the exponential
    schedule wins, so a small hint cannot defeat backoff under sustained load.
    """

    def __init__(self, fallback: wait_base) -> None:
        """Initialise the strategy.

        Args:
            fallback: The wait strategy used when no hint is available.
        """
        self._fallback = fallback

    def __call__(self, retry_state: RetryCallState) -> float:
        """Compute the delay before the next attempt.

        Args:
            retry_state: Tenacity's state for the failed attempt.

        Returns:
            The number of seconds to sleep.
        """
        base = self._fallback(retry_state)
        outcome = retry_state.outcome
        if outcome is None or not outcome.failed:
            return base
        hint = getattr(outcome.exception(), "retry_after", None)
        if isinstance(hint, (int, float)) and hint > base:
            return float(hint)
        return base


class AsyncRateLimiter:
    """An asyncio token-bucket limiter enforcing a requests-per-minute quota.

    Tokens accrue continuously at ``rpm / 60`` per second up to a burst
    capacity, so short bursts are permitted while the long-run average stays
    within quota. Acquisition is serialised by a lock, which also makes waiters
    approximately fair (FIFO).

    A ``None`` or non-positive rate disables limiting entirely, making the
    limiter a zero-cost no-op that callers can always invoke unconditionally.
    """

    def __init__(self, requests_per_minute: float | None, *, burst: int | None = None) -> None:
        """Initialise the bucket.

        Args:
            requests_per_minute: Sustained request quota, or ``None`` to disable.
            burst: Maximum tokens the bucket may hold. Defaults to one second's
                worth of tokens (minimum 1), which smooths traffic without
                allowing a large thundering herd on start-up.
        """
        self._enabled = requests_per_minute is not None and requests_per_minute > 0
        self._rate_per_second = (requests_per_minute or 0.0) / 60.0
        self._capacity = float(burst) if burst is not None else max(1.0, self._rate_per_second)
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        """Whether rate limiting is active.

        Returns:
            ``True`` when a positive quota was configured.
        """
        return self._enabled

    async def acquire(self, tokens: float = 1.0) -> float:
        """Consume tokens, sleeping until the bucket can satisfy the request.

        Args:
            tokens: Number of tokens to consume; one per request by default.

        Returns:
            The number of seconds spent waiting, useful for latency attribution.

        Raises:
            ValueError: If ``tokens`` exceeds the bucket capacity, which would
                otherwise wait forever.
        """
        if not self._enabled:
            return 0.0
        if tokens > self._capacity:
            msg = f"requested {tokens} tokens exceeds bucket capacity {self._capacity}"
            raise ValueError(msg)

        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                delay = deficit / self._rate_per_second
                waited += delay
                await asyncio.sleep(delay)

    def _refill(self) -> None:
        """Add tokens accrued since the last refill, capped at capacity."""
        now = time.monotonic()
        elapsed = now - self._updated_at
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)
            self._updated_at = now

    async def __aenter__(self) -> AsyncRateLimiter:
        """Acquire a single token on context entry.

        Returns:
            This limiter.
        """
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release nothing; the bucket refills on a timer.

        Args:
            exc_type: Exception type, if the block raised.
            exc: Exception instance, if the block raised.
            tb: Traceback, if the block raised.
        """
        return
