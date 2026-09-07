"""Orchestration of multiple metrics across a dataset.

:class:`EvaluationSuite` is the engine's entry point. It owns metric
construction, bounded-concurrency scheduling, optional guardrail execution,
aggregation and report assembly.

Concurrency is capped at the *sample* level (``max_concurrency``) while metrics
within a sample run in parallel. This gives predictable peak load -- at most
``max_concurrency x len(metrics)`` in-flight judge calls -- which the LLM
client's own semaphore and rate limiter then smooth into the provider's quota.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone

from ..config import EvaluationSettings, Settings
from ..guardrails.pipeline import GuardrailPipeline
from ..llm.base import LLMClient
from ..logging_config import get_logger, request_context
from ..models import (
    AggregateMetric,
    EvaluationReport,
    MetricName,
    RAGSample,
    RunConfig,
    SampleResult,
    SingleMetricResult,
)
from .answer_relevance import AnswerRelevanceEvaluator
from .base import BaseEvaluator
from .consistency import ConsistencyEvaluator
from .context_precision import ContextPrecisionEvaluator
from .context_recall import ContextRecallEvaluator
from .faithfulness import FaithfulnessEvaluator

__all__ = ["EVALUATOR_REGISTRY", "EvaluationSuite"]

_log = get_logger(__name__)

#: Maps each metric to the evaluator class implementing it.
EVALUATOR_REGISTRY: dict[MetricName, type[BaseEvaluator]] = {
    MetricName.CONTEXT_PRECISION: ContextPrecisionEvaluator,
    MetricName.CONTEXT_RECALL: ContextRecallEvaluator,
    MetricName.FAITHFULNESS: FaithfulnessEvaluator,
    MetricName.ANSWER_RELEVANCE: AnswerRelevanceEvaluator,
    MetricName.CONSISTENCY: ConsistencyEvaluator,
}


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Returns:
        The current UTC time.
    """
    return datetime.now(timezone.utc)


class EvaluationSuite:
    """Runs a configured set of metrics over samples and aggregates the results.

    Example:
        >>> suite = EvaluationSuite(llm_client, settings)  # doctest: +SKIP
        >>> report = await suite.aevaluate_dataset(samples, config)  # doctest: +SKIP
        >>> report.aggregates[MetricName.FAITHFULNESS].mean  # doctest: +SKIP
    """

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings,
        *,
        guardrails: GuardrailPipeline | None = None,
        evaluators: Sequence[BaseEvaluator] | None = None,
    ) -> None:
        """Initialise the suite.

        Args:
            llm: The evaluator LLM shared by every metric.
            settings: The full application settings.
            guardrails: Guardrail pipeline to run alongside metrics. When
                ``None`` and guardrails are enabled in settings, one is built
                from configuration; pass an explicit pipeline to override.
            evaluators: Explicit evaluator instances, bypassing the registry.
                Primarily used by tests and by callers registering custom
                metrics.
        """
        self._llm = llm
        self._settings = settings
        self._eval_settings: EvaluationSettings = settings.evaluation
        self._evaluators = list(
            evaluators if evaluators is not None else self._build_evaluators(llm, settings)
        )
        if guardrails is not None:
            self._guardrails: GuardrailPipeline | None = guardrails
        elif settings.guardrails.enabled:
            self._guardrails = GuardrailPipeline.from_settings(settings, llm=llm)
        else:
            self._guardrails = None

    @staticmethod
    def _build_evaluators(llm: LLMClient, settings: Settings) -> list[BaseEvaluator]:
        """Instantiate evaluators for the configured metrics.

        Args:
            llm: The evaluator LLM.
            settings: The full application settings.

        Returns:
            One evaluator per configured metric, in configuration order.
        """
        return [
            EVALUATOR_REGISTRY[metric](llm, settings.evaluation)
            for metric in settings.evaluation.metrics
            if metric in EVALUATOR_REGISTRY
        ]

    @property
    def metrics(self) -> list[MetricName]:
        """The metrics this suite will compute.

        Returns:
            The metric names, in execution order.
        """
        return [evaluator.metric for evaluator in self._evaluators]

    # ------------------------------------------------------------------ #
    # Single sample
    # ------------------------------------------------------------------ #

    async def aevaluate_sample(self, sample: RAGSample) -> SampleResult:
        """Evaluate every configured metric for one sample.

        Metrics run concurrently and independently; a failure in one is
        recorded on that metric's result and does not affect the others.

        Args:
            sample: The sample to evaluate.

        Returns:
            The consolidated per-sample result.
        """
        started = time.perf_counter()

        guardrail_task = None
        if self._guardrails is not None:
            guardrail_task = asyncio.ensure_future(self._guardrails.acheck_sample(sample))

        metric_results = await asyncio.gather(
            *(evaluator.aevaluate(sample) for evaluator in self._evaluators)
        )

        inbound = None
        outbound = None
        if guardrail_task is not None:
            inbound, outbound = await guardrail_task

        elapsed_ms = (time.perf_counter() - started) * 1000
        return SampleResult(
            sample_id=sample.sample_id,
            metrics={result.metric: result for result in metric_results},
            inbound_guardrail=inbound,
            outbound_guardrail=outbound,
            latency_ms=elapsed_ms,
            metadata=dict(sample.metadata),
        )

    def evaluate_sample(self, sample: RAGSample) -> SampleResult:
        """Synchronous facade over :meth:`aevaluate_sample`.

        Args:
            sample: The sample to evaluate.

        Returns:
            The consolidated per-sample result.

        Raises:
            RuntimeError: If called from within a running event loop.
        """
        from ..llm.base import _run_sync

        return _run_sync(self.aevaluate_sample(sample))

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #

    async def aevaluate_dataset(
        self,
        samples: Iterable[RAGSample],
        config: RunConfig | None = None,
        *,
        run_id: str | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> EvaluationReport:
        """Evaluate a dataset with bounded concurrency and aggregate the results.

        Args:
            samples: The dataset to evaluate. Fully materialised before the run
                so that ``sample_count`` and progress reporting are accurate.
            config: The RAG configuration under test, recorded in the report.
            run_id: Correlation id for this run; generated when omitted.
            progress: Optional callback invoked as ``(completed, total)`` after
                each sample finishes. Exceptions raised by the callback are
                logged and swallowed so reporting cannot abort a run.

        Returns:
            The aggregated evaluation report.

        Raises:
            Exception: Propagated from a sample only when ``fail_fast`` is set;
                otherwise per-sample failures are captured in the report.
        """
        materialised = list(samples)
        run_config = config or RunConfig(name="default")
        total = len(materialised)

        with request_context(run_id=run_id or "", config_name=run_config.name) as correlation_id:
            effective_run_id = run_id or correlation_id
            started_at = _utcnow()
            # Snapshot usage so a multi-configuration sweep sharing one client
            # attributes only this run's calls and tokens to this report.
            calls_before = self._llm.call_count
            tokens_before = self._llm.usage.total_tokens
            cost_before = self._llm.estimated_cost_usd() or 0.0
            _log.info(
                "evaluation.started",
                run_id=effective_run_id,
                config_name=run_config.name,
                sample_count=total,
                metrics=[m.value for m in self.metrics],
                max_concurrency=self._eval_settings.max_concurrency,
            )

            semaphore = asyncio.Semaphore(self._eval_settings.max_concurrency)
            completed = 0
            completion_lock = asyncio.Lock()

            async def _run_one(sample: RAGSample) -> SampleResult:
                """Evaluate one sample under the concurrency cap.

                Args:
                    sample: The sample to evaluate.

                Returns:
                    The per-sample result.
                """
                nonlocal completed
                async with semaphore:
                    result = await self.aevaluate_sample(sample)
                async with completion_lock:
                    completed += 1
                    current = completed
                if progress is not None:
                    try:
                        progress(current, total)
                    except Exception:  # noqa: BLE001 - reporting must not fail a run
                        _log.warning("evaluation.progress_callback_failed", exc_info=True)
                return result

            gathered = await asyncio.gather(
                *(_run_one(sample) for sample in materialised),
                return_exceptions=not self._eval_settings.fail_fast,
            )

            sample_results: list[SampleResult] = []
            for sample, outcome in zip(materialised, gathered, strict=True):
                if isinstance(outcome, BaseException):
                    _log.error(
                        "evaluation.sample_failed",
                        sample_id=sample.sample_id,
                        error_type=type(outcome).__name__,
                        error=str(outcome)[:500],
                    )
                    sample_results.append(self._failed_sample_result(sample, outcome))
                else:
                    sample_results.append(outcome)

            completed_at = _utcnow()
            report = self._build_report(
                run_id=effective_run_id,
                config=run_config,
                sample_results=sample_results,
                started_at=started_at,
                completed_at=completed_at,
                calls_before=calls_before,
                tokens_before=tokens_before,
                cost_before=cost_before,
            )
            _log.info(
                "evaluation.completed",
                run_id=effective_run_id,
                config_name=run_config.name,
                sample_count=report.sample_count,
                duration_seconds=round(report.duration_seconds, 2),
                failure_rate=round(report.failure_rate, 4),
                blocked_samples=report.blocked_samples,
                llm_calls=report.llm_calls,
                total_tokens=report.total_tokens,
                aggregates={
                    metric.value: None if agg.mean is None else round(agg.mean, 4)
                    for metric, agg in report.aggregates.items()
                },
            )
            return report

    def evaluate_dataset(
        self,
        samples: Iterable[RAGSample],
        config: RunConfig | None = None,
        *,
        run_id: str | None = None,
    ) -> EvaluationReport:
        """Synchronous facade over :meth:`aevaluate_dataset`.

        Args:
            samples: The dataset to evaluate.
            config: The RAG configuration under test.
            run_id: Correlation id for this run.

        Returns:
            The aggregated evaluation report.

        Raises:
            RuntimeError: If called from within a running event loop.
        """
        from ..llm.base import _run_sync

        return _run_sync(self.aevaluate_dataset(samples, config, run_id=run_id))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _failed_sample_result(self, sample: RAGSample, error: BaseException) -> SampleResult:
        """Build a result marking every metric as failed for one sample.

        Reached only when :meth:`aevaluate_sample` itself raises -- for example
        on cancellation -- since individual metric failures are already
        contained by :meth:`BaseEvaluator.aevaluate`.

        Args:
            sample: The sample that failed.
            error: The raised exception.

        Returns:
            A result whose metrics all carry the error.
        """
        message = f"{type(error).__name__}: {error}"
        return SampleResult(
            sample_id=sample.sample_id,
            metrics={
                evaluator.metric: SingleMetricResult(
                    metric=evaluator.metric, sample_id=sample.sample_id, error=message
                )
                for evaluator in self._evaluators
            },
            metadata=dict(sample.metadata),
        )

    def _build_report(
        self,
        *,
        run_id: str,
        config: RunConfig,
        sample_results: Sequence[SampleResult],
        started_at: datetime,
        completed_at: datetime,
        calls_before: int = 0,
        tokens_before: int = 0,
        cost_before: float = 0.0,
    ) -> EvaluationReport:
        """Aggregate per-sample results into a report.

        Args:
            run_id: Correlation id for the run.
            config: The RAG configuration under test.
            sample_results: All per-sample results.
            started_at: Run start timestamp.
            completed_at: Run completion timestamp.
            calls_before: Client call count captured before the run, so usage is
                attributed per run rather than per client lifetime.
            tokens_before: Client token count captured before the run.
            cost_before: Estimated client cost captured before the run.

        Returns:
            The assembled report.
        """
        aggregates: dict[MetricName, AggregateMetric] = {}
        total_evaluations = 0
        total_failures = 0

        for metric in self.metrics:
            per_metric = [
                result.metrics[metric] for result in sample_results if metric in result.metrics
            ]
            aggregates[metric] = AggregateMetric.from_results(
                metric, per_metric, threshold=self._eval_settings.threshold_for(metric)
            )
            total_evaluations += len(per_metric)
            total_failures += sum(1 for r in per_metric if r.error is not None)

        guardrail_triggers: dict[str, int] = {}
        blocked = 0
        for result in sample_results:
            if result.blocked:
                blocked += 1
            for report in (result.inbound_guardrail, result.outbound_guardrail):
                if report is None:
                    continue
                for key, count in report.trigger_counts().items():
                    guardrail_triggers[key] = guardrail_triggers.get(key, 0) + count

        return EvaluationReport(
            run_id=run_id,
            config_name=config.name,
            config_params=dict(config.params),
            dataset_uri=config.dataset_uri,
            judge_model=config.judge_model or self._llm.model_id,
            started_at=started_at,
            completed_at=completed_at,
            sample_count=len(sample_results),
            aggregates=aggregates,
            sample_results=(
                list(sample_results) if self._eval_settings.include_samples_in_report else []
            ),
            guardrail_triggers=guardrail_triggers,
            blocked_samples=blocked,
            failure_rate=(total_failures / total_evaluations) if total_evaluations else 0.0,
            llm_calls=max(0, self._llm.call_count - calls_before),
            total_tokens=max(0, self._llm.usage.total_tokens - tokens_before),
            estimated_cost_usd=_cost_delta(self._llm.estimated_cost_usd(), cost_before),
        )


def _cost_delta(current: float | None, before: float) -> float | None:
    """Compute this run's share of the client's cumulative estimated cost.

    Args:
        current: The client's cumulative cost estimate, or ``None`` when no
            pricing is configured.
        before: The cumulative estimate captured before the run started.

    Returns:
        The non-negative delta, or ``None`` when pricing is unavailable.
    """
    if current is None:
        return None
    return max(0.0, current - before)
