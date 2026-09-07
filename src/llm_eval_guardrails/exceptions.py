"""Exception hierarchy for the LLM evaluation and guardrails framework.

All framework errors derive from :class:`FrameworkError` so that callers can
distinguish framework failures from arbitrary runtime errors without catching
bare ``Exception``.
"""

from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "DatasetError",
    "EvaluationError",
    "FrameworkError",
    "GuardrailTripped",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponseFormatError",
    "LLMTransientError",
    "StorageError",
]


class FrameworkError(Exception):
    """Base class for every error raised by this framework."""


class ConfigurationError(FrameworkError):
    """Raised when settings are missing, malformed, or mutually inconsistent."""


class LLMError(FrameworkError):
    """Base class for failures originating from an evaluator LLM provider."""


class LLMTransientError(LLMError):
    """A retryable provider failure (throttling, 5xx, timeout, connection reset).

    Attributes:
        retry_after: Provider-supplied backoff hint in seconds. When present and
            larger than the computed exponential delay, the retry policy honours
            it instead.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        """Initialise the error.

        Args:
            message: Human-readable failure summary.
            retry_after: Optional provider backoff hint, in seconds.
        """
        super().__init__(message)
        self.retry_after = retry_after


class LLMRateLimitError(LLMTransientError):
    """The provider explicitly rejected the call because of rate limiting."""


class LLMResponseFormatError(LLMError):
    """The provider returned a payload that could not be parsed into the expected schema.

    This is deliberately *not* a subclass of :class:`LLMTransientError`; the
    judge layer retries it separately with a repair prompt rather than blindly
    re-issuing the identical request.
    """


class EvaluationError(FrameworkError):
    """Raised when a metric cannot be computed for a sample."""


class DatasetError(FrameworkError):
    """Raised when an evaluation dataset is unreadable or malformed."""


class StorageError(FrameworkError):
    """Raised when reading from or writing to object storage fails."""


class GuardrailTripped(FrameworkError):
    """Raised by ``enforce``-style helpers when a blocking guardrail fires.

    Attributes:
        report: The full guardrail report that triggered the block.
    """

    def __init__(self, message: str, report: object) -> None:
        """Initialise the error.

        Args:
            message: Human-readable summary of the violation.
            report: The originating guardrail report object.
        """
        super().__init__(message)
        self.report = report
