"""Environment-driven configuration for the evaluation and guardrails framework.

Every setting is overridable through environment variables prefixed with
``LEG_`` (nested settings use ``__`` as the delimiter), which keeps container
deployments declarative::

    LEG_LLM__PROVIDER=bedrock
    LEG_LLM__MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
    LEG_GUARDRAILS__BLOCK_ON_PII=true
    LEG_EVALUATION__MAX_CONCURRENCY=16
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from .exceptions import ConfigurationError
from .models import MetricName

__all__ = [
    "AWSSettings",
    "EvaluationSettings",
    "GuardrailSettings",
    "LLMProvider",
    "LLMSettings",
    "Settings",
    "get_settings",
    "reset_settings_cache",
]


class LLMProvider(str, Enum):
    """Supported evaluator-LLM backends."""

    BEDROCK = "bedrock"
    OPENAI = "openai"

    def __str__(self) -> str:
        """Return the raw provider identifier.

        Returns:
            The string value of the enum member.
        """
        return self.value


class _Section(BaseSettings):
    """Base class for nested settings sections."""

    model_config = SettingsConfigDict(extra="ignore", validate_assignment=True)


class LLMSettings(_Section):
    """Configuration for the evaluator LLM (LLM-as-a-judge)."""

    provider: LLMProvider | str = Field(
        default=LLMProvider.BEDROCK,
        description=(
            "Which provider backs the judge. Built-in values are validated against "
            "``LLMProvider``; any other string is resolved against the runtime "
            "registry by ``build_llm_client``, which is what makes "
            "``register_provider`` a usable extension point."
        ),
    )
    model_id: str = Field(
        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        description="Provider-specific model identifier.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; 0.0 maximises judge reproducibility.",
    )
    max_tokens: int = Field(
        default=2048, gt=0, le=32_000, description="Maximum tokens in a judge response."
    )
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling mass.")
    timeout_seconds: float = Field(
        default=60.0, gt=0.0, description="Per-request timeout for judge calls."
    )
    max_attempts: int = Field(
        default=5, ge=1, le=10, description="Total attempts per call, including the first."
    )
    initial_backoff_seconds: float = Field(
        default=0.5, gt=0.0, description="First retry delay for exponential backoff."
    )
    max_backoff_seconds: float = Field(
        default=20.0, gt=0.0, description="Ceiling for exponential backoff delays."
    )
    requests_per_minute: float | None = Field(
        default=None,
        gt=0.0,
        description="Client-side rate limit; ``None`` disables the limiter.",
    )
    max_in_flight: int = Field(
        default=8, ge=1, le=256, description="Maximum concurrent in-flight judge requests."
    )
    api_key: SecretStr | None = Field(
        default=None, description="Provider API key; unused by Bedrock (which uses IAM)."
    )
    base_url: str | None = Field(
        default=None, description="Override the provider endpoint (proxies, VPC endpoints)."
    )
    input_cost_per_1k: float | None = Field(
        default=None, ge=0.0, description="USD per 1K input tokens, for cost estimation."
    )
    output_cost_per_1k: float | None = Field(
        default=None, ge=0.0, description="USD per 1K output tokens, for cost estimation."
    )

    @model_validator(mode="after")
    def _validate_backoff(self) -> Self:
        """Ensure the backoff ceiling is not below the initial delay.

        Returns:
            The validated settings.

        Raises:
            ValueError: If ``max_backoff_seconds`` is below the initial delay.
        """
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            msg = "max_backoff_seconds must be >= initial_backoff_seconds"
            raise ValueError(msg)
        return self


class AWSSettings(_Section):
    """AWS region, credentials profile and S3 locations."""

    region: str = Field(default="us-east-1", description="AWS region for Bedrock and S3.")
    profile: str | None = Field(
        default=None, description="Named credentials profile; omit to use the default chain."
    )
    endpoint_url: str | None = Field(
        default=None, description="Override the S3 endpoint (LocalStack, VPC endpoints)."
    )
    report_bucket: str | None = Field(
        default=None, description="Default S3 bucket for evaluation reports."
    )
    report_prefix: str = Field(
        default="llm-eval/reports", description="Key prefix under which reports are written."
    )
    server_side_encryption: Literal["AES256", "aws:kms"] | None = Field(
        default="AES256", description="SSE mode applied to uploaded objects."
    )
    kms_key_id: str | None = Field(
        default=None, description="KMS key id, required when SSE is ``aws:kms``."
    )
    max_pool_connections: int = Field(
        default=32, ge=1, description="botocore connection pool size."
    )

    @model_validator(mode="after")
    def _validate_kms(self) -> Self:
        """Require a KMS key whenever KMS encryption is selected.

        Returns:
            The validated settings.

        Raises:
            ValueError: If ``aws:kms`` is selected without a key id.
        """
        if self.server_side_encryption == "aws:kms" and not self.kms_key_id:
            msg = "kms_key_id is required when server_side_encryption is 'aws:kms'"
            raise ValueError(msg)
        return self


class EvaluationSettings(_Section):
    """Concurrency, metric selection and pass thresholds."""

    metrics: list[MetricName] = Field(
        default_factory=lambda: list(MetricName),
        description="Metrics to compute; order is preserved in reports.",
    )
    max_concurrency: int = Field(
        default=8, ge=1, le=128, description="Maximum samples evaluated in parallel."
    )
    thresholds: dict[MetricName, float] = Field(
        default_factory=lambda: {
            MetricName.CONTEXT_PRECISION: 0.70,
            MetricName.CONTEXT_RECALL: 0.70,
            MetricName.FAITHFULNESS: 0.80,
            MetricName.ANSWER_RELEVANCE: 0.70,
            MetricName.CONSISTENCY: 0.75,
        },
        description="Per-metric pass thresholds used for pass-rate reporting.",
    )
    consistency_samples: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of resampled answers compared by the consistency metric.",
    )
    consistency_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature used when resampling answers for consistency.",
    )
    max_context_chars: int = Field(
        default=4000, gt=0, description="Per-chunk truncation budget in judge prompts."
    )
    fail_fast: bool = Field(default=False, description="Abort the run on the first metric failure.")
    include_samples_in_report: bool = Field(
        default=True, description="Embed per-sample detail in the written report."
    )
    max_failure_rate: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Run exits non-zero when the metric failure rate exceeds this.",
    )

    @field_validator("metrics")
    @classmethod
    def _dedupe_metrics(cls, value: list[MetricName]) -> list[MetricName]:
        """Remove duplicate metrics while preserving order.

        Args:
            value: The configured metric list.

        Returns:
            The deduplicated list.

        Raises:
            ValueError: If no metrics are configured.
        """
        if not value:
            msg = "at least one metric must be configured"
            raise ValueError(msg)
        seen: dict[MetricName, None] = {}
        for metric in value:
            seen.setdefault(metric, None)
        return list(seen)

    def threshold_for(self, metric: MetricName) -> float | None:
        """Return the configured pass threshold for a metric.

        Args:
            metric: The metric to look up.

        Returns:
            The threshold, or ``None`` when the metric has none.
        """
        return self.thresholds.get(metric)


class GuardrailSettings(_Section):
    """Inbound and outbound guardrail policy."""

    enabled: bool = Field(default=True, description="Master switch for all guardrails.")
    block_on_pii: bool = Field(
        default=False,
        description="Block prompts containing PII; when ``False``, PII is redacted instead.",
    )
    pii_entities: list[str] = Field(
        default_factory=list,
        description="Restrict PII detection to these entity types; empty means all.",
    )
    pii_redaction_token: str = Field(
        default="[REDACTED:{entity_type}]",
        description="Redaction template; ``{entity_type}`` is substituted.",
    )
    use_presidio: bool = Field(
        default=False,
        description="Augment regex detection with Presidio NER when the extra is installed.",
    )
    injection_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Heuristic injection score at or above which a prompt is blocked.",
    )
    max_prompt_chars: int = Field(
        default=20_000, gt=0, description="Reject prompts longer than this many characters."
    )
    blocked_topics: list[str] = Field(
        default_factory=list,
        description="Case-insensitive phrases that block a prompt or response outright.",
    )
    grounding_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum supported-claim ratio before a response is flagged.",
    )
    block_unsupported_claims: bool = Field(
        default=False,
        description="Block (rather than flag) responses below ``grounding_threshold``.",
    )
    use_llm_for_outbound: bool = Field(
        default=True,
        description="Use the judge LLM for claim-level grounding checks on responses.",
    )
    fail_closed: bool = Field(
        default=True,
        description="Treat a guardrail execution error as a block rather than an allow.",
    )

    @field_validator("pii_redaction_token")
    @classmethod
    def _validate_redaction_token(cls, value: str) -> str:
        """Ensure the redaction template is non-empty and safely formattable.

        Args:
            value: The configured template.

        Returns:
            The validated template.

        Raises:
            ValueError: If the template is empty or contains unknown fields.
        """
        if not value.strip():
            msg = "pii_redaction_token must not be empty"
            raise ValueError(msg)
        try:
            value.format(entity_type="test")
        except (KeyError, IndexError) as exc:
            msg = "pii_redaction_token supports only the {entity_type} placeholder"
            raise ValueError(msg) from exc
        return value


class Settings(BaseSettings):
    """Top-level application settings assembled from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="LEG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    environment: str = Field(default="local", description="Deployment environment name.")
    log_level: str = Field(default="INFO", description="Root log level.")
    json_logs: bool | None = Field(
        default=None, description="Force JSON logging; ``None`` auto-detects a TTY."
    )
    llm: LLMSettings = Field(default_factory=LLMSettings, description="Evaluator LLM settings.")
    aws: AWSSettings = Field(default_factory=AWSSettings, description="AWS integration settings.")
    evaluation: EvaluationSettings = Field(
        default_factory=EvaluationSettings, description="Evaluation engine settings."
    )
    guardrails: GuardrailSettings = Field(
        default_factory=GuardrailSettings, description="Guardrail policy settings."
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalise and validate the log level name.

        Args:
            value: The configured level name.

        Returns:
            The upper-cased level name.

        Raises:
            ValueError: If the name is not a standard logging level.
        """
        normalised = value.strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalised not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}"
            raise ValueError(msg)
        return normalised

    @model_validator(mode="after")
    def _validate_provider_credentials(self) -> Self:
        """Verify provider-specific prerequisites are satisfied.

        Bedrock authenticates through the standard AWS credential chain, so no
        API key is required. OpenAI requires an API key, which may also come
        from the provider SDK's own ``OPENAI_API_KEY`` environment variable;
        that fallback is resolved lazily by the client, so it is not enforced
        here.

        Returns:
            The validated settings.
        """
        return self

    def describe(self) -> dict[str, Any]:
        """Render a secret-free snapshot of the settings for logging.

        Returns:
            A nested mapping with every :class:`~pydantic.SecretStr` masked.
        """
        payload = self.model_dump(mode="json")
        llm_section = payload.get("llm")
        if isinstance(llm_section, dict) and llm_section.get("api_key") is not None:
            llm_section["api_key"] = "***"
        return payload


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the process-wide settings.

    Returns:
        The validated settings singleton.

    Raises:
        ConfigurationError: If the environment does not produce valid settings.
    """
    try:
        return Settings()
    except Exception as exc:
        msg = f"invalid configuration: {exc}"
        raise ConfigurationError(msg) from exc


def reset_settings_cache() -> None:
    """Clear the cached settings singleton.

    Intended for tests and for long-lived processes that reload configuration
    after an environment change.
    """
    get_settings.cache_clear()
