"""Pydantic domain models shared across evaluators, guardrails and workflows.

These models are the framework's public contract. They are intentionally
strict (``extra="forbid"``) so that typos in datasets or judge responses fail
loudly at the boundary rather than silently degrading a metric.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from typing_extensions import Self

__all__ = [
    "AggregateMetric",
    "Claim",
    "ClaimVerdict",
    "EvaluationReport",
    "GuardrailAction",
    "GuardrailFinding",
    "GuardrailReport",
    "GuardrailStage",
    "JudgeVerdict",
    "MetricName",
    "PIIEntity",
    "PIIEntityType",
    "RAGSample",
    "RetrievedDocument",
    "RunConfig",
    "SampleResult",
    "Score",
    "Severity",
    "SingleMetricResult",
]

Score = Annotated[float, Field(ge=0.0, le=1.0)]
"""A normalised score in the closed interval [0, 1]."""


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Returns:
        The current UTC time.
    """
    return datetime.now(timezone.utc)


class _Base(BaseModel):
    """Shared base model applying strict validation configuration."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        """Discard computed fields so serialised models can be re-validated.

        ``model_dump`` emits computed fields (``duration_seconds``, ``allowed``,
        ...), but ``extra="forbid"`` would reject them on the way back in. That
        would make a published report impossible to load for a regression
        comparison. Dropping exactly the computed names keeps round-tripping
        lossless while still rejecting genuine typos.

        Args:
            data: The raw input passed to validation.

        Returns:
            The input with computed-field keys removed, when it is a mapping.
        """
        computed = cls.model_computed_fields
        if computed and isinstance(data, dict) and any(key in data for key in computed):
            return {key: value for key, value in data.items() if key not in computed}
        return data


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class MetricName(str, Enum):
    """Canonical identifiers for the metrics this framework computes."""

    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONSISTENCY = "consistency"

    def __str__(self) -> str:
        """Return the raw metric identifier.

        Returns:
            The string value of the enum member.
        """
        return self.value


class Severity(str, Enum):
    """Severity of a guardrail finding, ordered from least to most serious."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Return the ordinal rank of this severity.

        Returns:
            ``0`` for ``INFO`` through ``4`` for ``CRITICAL``.
        """
        return _SEVERITY_ORDER[self]

    def __str__(self) -> str:
        """Return the raw severity identifier.

        Returns:
            The string value of the enum member.
        """
        return self.value


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class GuardrailAction(str, Enum):
    """The enforcement decision produced by a guardrail pipeline."""

    ALLOW = "allow"
    FLAG = "flag"
    REDACT = "redact"
    BLOCK = "block"

    @property
    def rank(self) -> int:
        """Return the ordinal severity of this action.

        Returns:
            ``0`` for ``ALLOW`` through ``3`` for ``BLOCK``.
        """
        return _ACTION_ORDER[self]

    def __str__(self) -> str:
        """Return the raw action identifier.

        Returns:
            The string value of the enum member.
        """
        return self.value


_ACTION_ORDER: dict[GuardrailAction, int] = {
    GuardrailAction.ALLOW: 0,
    GuardrailAction.FLAG: 1,
    GuardrailAction.REDACT: 2,
    GuardrailAction.BLOCK: 3,
}


class GuardrailStage(str, Enum):
    """Whether a guardrail inspects the inbound prompt or the outbound response."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"

    def __str__(self) -> str:
        """Return the raw stage identifier.

        Returns:
            The string value of the enum member.
        """
        return self.value


class PIIEntityType(str, Enum):
    """Categories of personally identifiable or secret information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    IP_ADDRESS = "ip_address"
    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"  # noqa: S105 - a category name, not a secret
    PRIVATE_KEY = "private_key"
    JWT = "jwt"
    PASSPORT = "passport"
    DATE_OF_BIRTH = "date_of_birth"
    PERSON = "person"
    LOCATION = "location"
    OTHER = "other"

    def __str__(self) -> str:
        """Return the raw entity-type identifier.

        Returns:
            The string value of the enum member.
        """
        return self.value


class ClaimVerdict(str, Enum):
    """Whether a single extracted claim is supported by the retrieved context."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"

    def __str__(self) -> str:
        """Return the raw verdict identifier.

        Returns:
            The string value of the enum member.
        """
        return self.value


# --------------------------------------------------------------------------- #
# Dataset / input models
# --------------------------------------------------------------------------- #


class RetrievedDocument(_Base):
    """A single chunk returned by the RAG retriever."""

    doc_id: str = Field(description="Stable identifier of the source chunk.")
    content: str = Field(min_length=1, description="Chunk text supplied to the generator.")
    score: float | None = Field(
        default=None, description="Raw retriever similarity score, if available."
    )
    rank: int | None = Field(
        default=None, ge=0, description="Zero-based position in the retrieved list."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary source metadata (uri, page, tenant)."
    )

    def truncated(self, max_chars: int) -> str:
        """Return the chunk content clipped to ``max_chars``.

        Args:
            max_chars: Maximum number of characters to retain. Non-positive
                values disable truncation.

        Returns:
            The content, suffixed with a truncation marker when clipped.
        """
        if max_chars <= 0 or len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + " ...[truncated]"


class RAGSample(_Base):
    """One question/answer/context triple to be evaluated.

    A sample is the atomic unit of evaluation. ``ground_truth`` is optional
    because reference-free metrics (faithfulness, answer relevance, context
    precision, consistency) do not require it; context recall does.
    """

    sample_id: str = Field(description="Unique identifier within the dataset.")
    question: str = Field(min_length=1, description="The end-user question.")
    answer: str = Field(default="", description="The RAG system's generated response.")
    contexts: list[RetrievedDocument] = Field(
        default_factory=list, description="Chunks retrieved for this question."
    )
    ground_truth: str | None = Field(
        default=None, description="Reference answer, required for context recall."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form sample metadata (tenant, locale, tags)."
    )

    @property
    def context_texts(self) -> list[str]:
        """Return the raw text of every retrieved chunk.

        Returns:
            The ``content`` of each context, in retrieval order.
        """
        return [c.content for c in self.contexts]

    def joined_context(self, max_chars_per_doc: int = 4000) -> str:
        """Render contexts as an enumerated block for judge prompts.

        Args:
            max_chars_per_doc: Per-chunk truncation budget.

        Returns:
            A newline-separated, 1-indexed listing of the contexts, or an
            explicit sentinel when no contexts were retrieved.
        """
        if not self.contexts:
            return "[NO CONTEXT RETRIEVED]"
        return "\n\n".join(
            f"[{i}] {doc.truncated(max_chars_per_doc)}"
            for i, doc in enumerate(self.contexts, start=1)
        )


class RunConfig(_Base):
    """A named RAG configuration being compared in a sweep.

    The framework does not execute the RAG pipeline itself; ``params`` is
    recorded verbatim in the report so results stay traceable to the retriever
    and generator settings that produced them.
    """

    name: str = Field(min_length=1, description="Human-readable configuration name.")
    description: str | None = Field(default=None, description="What this configuration varies.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary RAG parameters (top_k, embedder, model)."
    )
    dataset_uri: str | None = Field(
        default=None,
        description="Per-configuration dataset override (local path or s3:// URI).",
    )
    judge_model: str | None = Field(
        default=None, description="Override the evaluator LLM model for this configuration."
    )


# --------------------------------------------------------------------------- #
# Judge / evaluation models
# --------------------------------------------------------------------------- #


class Claim(_Base):
    """An atomic factual statement extracted from a generated answer."""

    text: str = Field(min_length=1, description="The self-contained claim.")
    verdict: ClaimVerdict = Field(description="Support status against the retrieved context.")
    supporting_context_ids: list[int] = Field(
        default_factory=list,
        description="1-based indices of contexts that support the claim.",
    )
    reasoning: str | None = Field(default=None, description="Judge rationale, one sentence.")

    @property
    def is_supported(self) -> bool:
        """Whether the claim is grounded in the retrieved context.

        Returns:
            ``True`` only when the verdict is :attr:`ClaimVerdict.SUPPORTED`.
        """
        return self.verdict is ClaimVerdict.SUPPORTED


class JudgeVerdict(_Base):
    """Normalised output of a single LLM-as-a-judge invocation."""

    score: Score = Field(description="Normalised metric score in [0, 1].")
    reasoning: str | None = Field(default=None, description="Concise judge rationale.")
    claims: list[Claim] = Field(
        default_factory=list, description="Per-claim breakdown, when the metric produces one."
    )
    raw: dict[str, Any] = Field(
        default_factory=dict, description="Unmodified parsed judge payload, for audit."
    )


class SingleMetricResult(_Base):
    """The outcome of evaluating one metric against one sample."""

    metric: MetricName = Field(description="Which metric was computed.")
    sample_id: str = Field(description="Identifier of the evaluated sample.")
    score: Score | None = Field(
        default=None,
        description="Score in [0, 1]; ``None`` when the metric failed or was skipped.",
    )
    passed: bool | None = Field(
        default=None, description="Score compared against the metric threshold, when configured."
    )
    reasoning: str | None = Field(default=None, description="Judge rationale for the score.")
    claims: list[Claim] = Field(default_factory=list, description="Per-claim breakdown, if any.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Wall-clock evaluation latency.")
    error: str | None = Field(
        default=None, description="Failure summary when the metric could not be computed."
    )
    skipped_reason: str | None = Field(
        default=None,
        description="Why the metric was deliberately not run (e.g. no ground truth).",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Evaluator-specific extras.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def succeeded(self) -> bool:
        """Whether a usable score was produced.

        Returns:
            ``True`` when a score is present and no error was recorded.
        """
        return self.error is None and self.score is not None


class PIIEntity(_Base):
    """One detected PII or secret span within a piece of text."""

    entity_type: PIIEntityType = Field(description="Category of the detected entity.")
    start: int = Field(ge=0, description="Inclusive start offset in the source text.")
    end: int = Field(ge=0, description="Exclusive end offset in the source text.")
    confidence: Score = Field(default=1.0, description="Detector confidence in [0, 1].")
    detector: str = Field(default="regex", description="Which detector produced the match.")

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        """Ensure the span is non-empty and correctly ordered.

        Returns:
            The validated model.

        Raises:
            ValueError: If ``end`` does not exceed ``start``.
        """
        if self.end <= self.start:
            msg = f"invalid span: end ({self.end}) must exceed start ({self.start})"
            raise ValueError(msg)
        return self


class GuardrailFinding(_Base):
    """A single policy observation produced by one guardrail check."""

    rule_id: str = Field(min_length=1, description="Stable identifier of the rule that fired.")
    stage: GuardrailStage = Field(description="Inbound or outbound.")
    severity: Severity = Field(description="How serious the finding is.")
    action: GuardrailAction = Field(description="Action recommended by this rule alone.")
    message: str = Field(min_length=1, description="Human-readable explanation.")
    score: Score | None = Field(
        default=None, description="Detector score, where the rule produces one."
    )
    entities: list[PIIEntity] = Field(default_factory=list, description="PII spans, for PII rules.")
    evidence: list[str] = Field(
        default_factory=list, description="Short quoted snippets supporting the finding."
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Rule-specific extras.")


class GuardrailReport(_Base):
    """The consolidated verdict of every guardrail run for one stage."""

    stage: GuardrailStage = Field(description="Which stage produced this report.")
    action: GuardrailAction = Field(
        default=GuardrailAction.ALLOW, description="Most severe action across all findings."
    )
    findings: list[GuardrailFinding] = Field(
        default_factory=list, description="Every finding raised, in detection order."
    )
    sanitized_text: str | None = Field(
        default=None, description="Redacted text when the action is ``redact``."
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="Wall-clock guardrail latency.")
    checked_at: datetime = Field(default_factory=_utcnow, description="UTC evaluation timestamp.")
    errors: list[str] = Field(
        default_factory=list, description="Names of checks that failed to execute."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed(self) -> bool:
        """Whether the content may proceed downstream.

        Returns:
            ``True`` for every action other than :attr:`GuardrailAction.BLOCK`.
        """
        return self.action is not GuardrailAction.BLOCK

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_severity(self) -> Severity | None:
        """The highest severity across all findings.

        Returns:
            The most severe finding's severity, or ``None`` when clean.
        """
        if not self.findings:
            return None
        return max((f.severity for f in self.findings), key=lambda s: s.rank)

    @property
    def triggered_rules(self) -> list[str]:
        """Rule identifiers that fired, deduplicated and ordered by first sighting.

        Returns:
            The list of distinct rule ids.
        """
        seen: dict[str, None] = {}
        for finding in self.findings:
            seen.setdefault(finding.rule_id, None)
        return list(seen)

    @classmethod
    def combine(
        cls,
        stage: GuardrailStage,
        findings: Sequence[GuardrailFinding],
        *,
        sanitized_text: str | None = None,
        latency_ms: float = 0.0,
        errors: Sequence[str] = (),
    ) -> GuardrailReport:
        """Build a report by escalating to the most severe recommended action.

        Args:
            stage: The stage the findings belong to.
            findings: All findings raised by the stage's checks.
            sanitized_text: Redacted text, when redaction was applied.
            latency_ms: Total stage latency in milliseconds.
            errors: Names of checks that raised.

        Returns:
            A report whose ``action`` is the maximum over all finding actions.
        """
        action = GuardrailAction.ALLOW
        for finding in findings:
            if finding.action.rank > action.rank:
                action = finding.action
        return cls(
            stage=stage,
            action=action,
            findings=list(findings),
            sanitized_text=sanitized_text,
            latency_ms=latency_ms,
            errors=list(errors),
        )

    def trigger_counts(self) -> Mapping[str, int]:
        """Count findings per rule for observability aggregation.

        Returns:
            A mapping of ``"<stage>:<rule_id>"`` to occurrence count.
        """
        counts: dict[str, int] = {}
        for finding in self.findings:
            key = f"{self.stage.value}:{finding.rule_id}"
            counts[key] = counts.get(key, 0) + 1
        return counts


class SampleResult(_Base):
    """All metric results plus guardrail reports for one sample."""

    sample_id: str = Field(description="Identifier of the evaluated sample.")
    config_name: str | None = Field(default=None, description="RAG configuration under test.")
    metrics: dict[MetricName, SingleMetricResult] = Field(
        default_factory=dict, description="Metric results keyed by metric name."
    )
    inbound_guardrail: GuardrailReport | None = Field(
        default=None, description="Prompt-side guardrail report, when guardrails ran."
    )
    outbound_guardrail: GuardrailReport | None = Field(
        default=None, description="Response-side guardrail report, when guardrails ran."
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="Total wall-clock time for sample.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Propagated sample metadata."
    )

    def score(self, metric: MetricName) -> float | None:
        """Look up the score for one metric.

        Args:
            metric: The metric to read.

        Returns:
            The score, or ``None`` when the metric is absent or failed.
        """
        result = self.metrics.get(metric)
        return result.score if result is not None else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blocked(self) -> bool:
        """Whether either guardrail stage produced a blocking decision.

        Returns:
            ``True`` when the inbound or outbound report blocked.
        """
        return any(
            report is not None and report.action is GuardrailAction.BLOCK
            for report in (self.inbound_guardrail, self.outbound_guardrail)
        )


# --------------------------------------------------------------------------- #
# Aggregation / reporting
# --------------------------------------------------------------------------- #


def _percentile(sorted_scores: Sequence[float], q: float) -> float:
    """Compute a percentile with linear interpolation.

    Args:
        sorted_scores: Ascending, non-empty sequence of scores.
        q: Quantile in [0, 1].

    Returns:
        The interpolated percentile value.
    """
    if len(sorted_scores) == 1:
        return sorted_scores[0]
    pos = q * (len(sorted_scores) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_scores) - 1)
    frac = pos - lower
    return sorted_scores[lower] * (1 - frac) + sorted_scores[upper] * frac


class AggregateMetric(_Base):
    """Descriptive statistics for one metric across a dataset."""

    metric: MetricName = Field(description="The aggregated metric.")
    count: int = Field(ge=0, description="Number of samples with a usable score.")
    failed: int = Field(default=0, ge=0, description="Number of samples that errored.")
    skipped: int = Field(default=0, ge=0, description="Number of samples deliberately skipped.")
    mean: float | None = Field(default=None, description="Arithmetic mean of scores.")
    median: float | None = Field(default=None, description="Median score.")
    stdev: float | None = Field(default=None, description="Sample standard deviation.")
    p10: float | None = Field(default=None, description="10th percentile score.")
    p90: float | None = Field(default=None, description="90th percentile score.")
    minimum: float | None = Field(default=None, description="Lowest observed score.")
    maximum: float | None = Field(default=None, description="Highest observed score.")
    pass_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of scored samples meeting threshold."
    )
    threshold: float | None = Field(default=None, description="Threshold used for ``pass_rate``.")

    @classmethod
    def from_results(
        cls,
        metric: MetricName,
        results: Sequence[SingleMetricResult],
        *,
        threshold: float | None = None,
    ) -> AggregateMetric:
        """Aggregate per-sample results into descriptive statistics.

        Samples that errored or were skipped are excluded from the statistics
        but counted separately, so a high mean over a handful of successes is
        never mistaken for a healthy run.

        Args:
            metric: The metric being aggregated.
            results: Per-sample results for this metric.
            threshold: Optional pass threshold used to compute ``pass_rate``.

        Returns:
            The populated aggregate.
        """
        scores = sorted(r.score for r in results if r.succeeded and r.score is not None)
        failed = sum(1 for r in results if r.error is not None)
        skipped = sum(1 for r in results if r.error is None and r.skipped_reason is not None)

        if not scores:
            return cls(metric=metric, count=0, failed=failed, skipped=skipped, threshold=threshold)

        pass_rate: float | None = None
        if threshold is not None:
            pass_rate = sum(1 for s in scores if s >= threshold) / len(scores)

        return cls(
            metric=metric,
            count=len(scores),
            failed=failed,
            skipped=skipped,
            mean=statistics.fmean(scores),
            median=statistics.median(scores),
            stdev=statistics.stdev(scores) if len(scores) > 1 else 0.0,
            p10=_percentile(scores, 0.10),
            p90=_percentile(scores, 0.90),
            minimum=scores[0],
            maximum=scores[-1],
            pass_rate=pass_rate,
            threshold=threshold,
        )


class EvaluationReport(_Base):
    """The aggregated result of evaluating one RAG configuration."""

    run_id: str = Field(description="Correlation id shared by all logs for this run.")
    config_name: str = Field(description="Name of the RAG configuration evaluated.")
    config_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters of the configuration under test."
    )
    dataset_uri: str | None = Field(default=None, description="Source of the evaluation dataset.")
    judge_model: str | None = Field(default=None, description="Evaluator LLM model identifier.")
    started_at: datetime = Field(description="UTC start timestamp.")
    completed_at: datetime = Field(description="UTC completion timestamp.")
    sample_count: int = Field(ge=0, description="Number of samples evaluated.")
    aggregates: dict[MetricName, AggregateMetric] = Field(
        default_factory=dict, description="Per-metric descriptive statistics."
    )
    sample_results: list[SampleResult] = Field(
        default_factory=list, description="Per-sample detail; omitted in summary reports."
    )
    guardrail_triggers: dict[str, int] = Field(
        default_factory=dict, description="Count of findings keyed by ``stage:rule_id``."
    )
    blocked_samples: int = Field(default=0, ge=0, description="Samples blocked by any guardrail.")
    failure_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of metric evaluations that errored."
    )
    llm_calls: int = Field(default=0, ge=0, description="Total judge invocations issued.")
    total_tokens: int = Field(default=0, ge=0, description="Total judge tokens consumed.")
    estimated_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Estimated judge cost, when pricing is configured."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the run.

        Returns:
            Seconds between ``started_at`` and ``completed_at``.
        """
        return (self.completed_at - self.started_at).total_seconds()

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        """Reject reports whose completion precedes their start.

        Returns:
            The validated model.

        Raises:
            ValueError: If ``completed_at`` is earlier than ``started_at``.
        """
        if self.completed_at < self.started_at:
            msg = "completed_at must not precede started_at"
            raise ValueError(msg)
        return self

    def summary_row(self) -> dict[str, Any]:
        """Flatten the report into a single row for cross-config comparison.

        Returns:
            A mapping of scalar values suitable for CSV or a Markdown table.
        """
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "config_name": self.config_name,
            "sample_count": self.sample_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "failure_rate": round(self.failure_rate, 4),
            "blocked_samples": self.blocked_samples,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
        }
        for metric, agg in self.aggregates.items():
            row[metric.value + "_mean"] = None if agg.mean is None else round(agg.mean, 4)
            if agg.pass_rate is not None:
                row[metric.value + "_pass_rate"] = round(agg.pass_rate, 4)
        return row

    def without_samples(self) -> EvaluationReport:
        """Return a copy with per-sample detail removed.

        Useful when publishing a compact summary alongside a full report.

        Returns:
            A shallow copy whose ``sample_results`` is empty.
        """
        return self.model_copy(update={"sample_results": []})
