"""Composition and enforcement of guardrail checks.

:class:`GuardrailPipeline` runs the inbound and outbound check chains and
reduces their findings into a single :class:`~llm_eval_guardrails.models.GuardrailReport`
per stage.

Execution order is not arbitrary. Fast checks run first, concurrently, and if
any of them blocks the pipeline short-circuits without invoking the slow
LLM-backed checks. A prompt-injection attempt is therefore rejected in
microseconds, and the judge budget is spent only on traffic that has already
passed the cheap tests.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from ..config import Settings
from ..exceptions import GuardrailTripped
from ..llm.base import LLMClient
from ..logging_config import get_logger
from ..models import (
    GuardrailAction,
    GuardrailFinding,
    GuardrailReport,
    GuardrailStage,
    RAGSample,
    RetrievedDocument,
)
from .base import CheckContext, GuardrailCheck
from .inbound import PIIGuardrailCheck, build_inbound_checks
from .outbound import build_outbound_checks
from .pii import PIIDetector, redact

__all__ = ["GuardrailPipeline"]

_log = get_logger(__name__)


class GuardrailPipeline:
    """Runs inbound and outbound guardrails and produces enforcement decisions.

    Example:
        >>> pipeline = GuardrailPipeline.from_settings(settings)  # doctest: +SKIP
        >>> report = pipeline.check_prompt("What is our refund policy?")  # doctest: +SKIP
        >>> report.allowed  # doctest: +SKIP
        True
    """

    def __init__(
        self,
        settings: Settings,
        *,
        inbound_checks: Sequence[GuardrailCheck],
        outbound_checks: Sequence[GuardrailCheck],
    ) -> None:
        """Initialise the pipeline with explicit check chains.

        Most callers should use :meth:`from_settings`; this constructor exists
        for tests and for deployments assembling a bespoke chain.

        Args:
            settings: The full application settings.
            inbound_checks: Checks applied to prompts.
            outbound_checks: Checks applied to responses.
        """
        self._settings = settings
        self._guardrail_settings = settings.guardrails
        self._inbound = list(inbound_checks)
        self._outbound = list(outbound_checks)

    @classmethod
    def from_settings(
        cls, settings: Settings, *, llm: LLMClient | None = None
    ) -> GuardrailPipeline:
        """Build a pipeline from configuration.

        A single :class:`~llm_eval_guardrails.guardrails.pii.PIIDetector` is
        shared by every check that needs one, which matters when Presidio is
        enabled: initialising its spaCy pipeline more than once would cost
        hundreds of megabytes and seconds of start-up time.

        Args:
            settings: The full application settings.
            llm: Judge LLM for the outbound grounding check. Without it, the
                pipeline runs fast checks only.

        Returns:
            The configured pipeline.
        """
        detector = PIIDetector(
            allowed_types=settings.guardrails.pii_entities,
            use_presidio=settings.guardrails.use_presidio,
        )
        return cls(
            settings,
            inbound_checks=build_inbound_checks(settings.guardrails, detector=detector),
            outbound_checks=build_outbound_checks(settings, llm=llm, detector=detector),
        )

    # ------------------------------------------------------------------ #
    # Inbound
    # ------------------------------------------------------------------ #

    async def acheck_prompt(
        self, prompt: str, *, sample_id: str | None = None, **metadata: object
    ) -> GuardrailReport:
        """Run inbound guardrails against a prompt.

        Args:
            prompt: The user prompt to inspect.
            sample_id: Identifier for log correlation.
            **metadata: Arbitrary context recorded on the check context.

        Returns:
            The consolidated inbound report. When the decision is
            ``REDACT``, ``sanitized_text`` holds the prompt safe to forward.
        """
        context = CheckContext(question=prompt, sample_id=sample_id, metadata=dict(metadata))
        return await self._run_stage(GuardrailStage.INBOUND, prompt, context, self._inbound)

    def check_prompt(
        self, prompt: str, *, sample_id: str | None = None, **metadata: object
    ) -> GuardrailReport:
        """Synchronous facade over :meth:`acheck_prompt`.

        Args:
            prompt: The user prompt to inspect.
            sample_id: Identifier for log correlation.
            **metadata: Arbitrary context recorded on the check context.

        Returns:
            The consolidated inbound report.

        Raises:
            RuntimeError: If called from within a running event loop.
        """
        from ..llm.base import _run_sync

        return _run_sync(self.acheck_prompt(prompt, sample_id=sample_id, **metadata))

    async def aenforce_prompt(self, prompt: str, *, sample_id: str | None = None) -> str:
        """Run inbound guardrails and return the prompt safe to forward.

        Args:
            prompt: The user prompt to inspect.
            sample_id: Identifier for log correlation.

        Returns:
            The original prompt, or its redacted form when redaction applied.

        Raises:
            GuardrailTripped: If the prompt was blocked.
        """
        report = await self.acheck_prompt(prompt, sample_id=sample_id)
        if report.action is GuardrailAction.BLOCK:
            msg = "inbound guardrail blocked the prompt: {rules}".format(
                rules=", ".join(report.triggered_rules)
            )
            raise GuardrailTripped(msg, report)
        return report.sanitized_text if report.sanitized_text is not None else prompt

    # ------------------------------------------------------------------ #
    # Outbound
    # ------------------------------------------------------------------ #

    async def acheck_response(
        self,
        response: str,
        *,
        question: str | None = None,
        contexts: Sequence[RetrievedDocument] | None = None,
        sample_id: str | None = None,
        **metadata: object,
    ) -> GuardrailReport:
        """Run outbound guardrails against a generated response.

        Args:
            response: The generated response to inspect.
            question: The originating question, used by the grounding check.
            contexts: The retrieved chunks the response must be grounded in.
            sample_id: Identifier for log correlation.
            **metadata: Arbitrary context recorded on the check context.

        Returns:
            The consolidated outbound report.
        """
        context = CheckContext(
            question=question,
            contexts=list(contexts or []),
            sample_id=sample_id,
            metadata=dict(metadata),
        )
        return await self._run_stage(GuardrailStage.OUTBOUND, response, context, self._outbound)

    def check_response(
        self,
        response: str,
        *,
        question: str | None = None,
        contexts: Sequence[RetrievedDocument] | None = None,
        sample_id: str | None = None,
        **metadata: object,
    ) -> GuardrailReport:
        """Synchronous facade over :meth:`acheck_response`.

        Args:
            response: The generated response to inspect.
            question: The originating question.
            contexts: The retrieved chunks.
            sample_id: Identifier for log correlation.
            **metadata: Arbitrary context recorded on the check context.

        Returns:
            The consolidated outbound report.

        Raises:
            RuntimeError: If called from within a running event loop.
        """
        from ..llm.base import _run_sync

        return _run_sync(
            self.acheck_response(
                response,
                question=question,
                contexts=contexts,
                sample_id=sample_id,
                **metadata,
            )
        )

    async def aenforce_response(
        self,
        response: str,
        *,
        question: str | None = None,
        contexts: Sequence[RetrievedDocument] | None = None,
        sample_id: str | None = None,
    ) -> str:
        """Run outbound guardrails and return the response safe to serve.

        Args:
            response: The generated response to inspect.
            question: The originating question.
            contexts: The retrieved chunks.
            sample_id: Identifier for log correlation.

        Returns:
            The original response, or its redacted form when redaction applied.

        Raises:
            GuardrailTripped: If the response was blocked.
        """
        report = await self.acheck_response(
            response, question=question, contexts=contexts, sample_id=sample_id
        )
        if report.action is GuardrailAction.BLOCK:
            msg = "outbound guardrail blocked the response: {rules}".format(
                rules=", ".join(report.triggered_rules)
            )
            raise GuardrailTripped(msg, report)
        return report.sanitized_text if report.sanitized_text is not None else response

    # ------------------------------------------------------------------ #
    # Combined
    # ------------------------------------------------------------------ #

    async def acheck_sample(self, sample: RAGSample) -> tuple[GuardrailReport, GuardrailReport]:
        """Run both stages for one evaluation sample.

        Stages run concurrently: unlike in production serving, the response
        already exists, so there is nothing to gain by gating the outbound
        check on the inbound result.

        Args:
            sample: The sample to inspect.

        Returns:
            A tuple of the inbound and outbound reports.
        """
        inbound, outbound = await asyncio.gather(
            self.acheck_prompt(sample.question, sample_id=sample.sample_id),
            self.acheck_response(
                sample.answer,
                question=sample.question,
                contexts=sample.contexts,
                sample_id=sample.sample_id,
            ),
        )
        return inbound, outbound

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _run_stage(
        self,
        stage: GuardrailStage,
        text: str,
        context: CheckContext,
        checks: Sequence[GuardrailCheck],
    ) -> GuardrailReport:
        """Run one stage's checks and reduce them to a single report.

        Args:
            stage: The stage being run.
            text: The text under inspection.
            context: Supporting context for the checks.
            checks: The checks to run.

        Returns:
            The consolidated report for this stage.
        """
        if not self._guardrail_settings.enabled or not checks:
            return GuardrailReport(stage=stage, action=GuardrailAction.ALLOW)

        started = time.perf_counter()
        fast = [c for c in checks if c.is_fast]
        slow = [c for c in checks if not c.is_fast]

        findings, errors = await self._gather(fast, text, context)

        # Short-circuit before paying for LLM-backed checks: the decision
        # cannot get any less severe than BLOCK, so the extra calls would only
        # add latency and cost.
        blocked_early = any(f.action is GuardrailAction.BLOCK for f in findings)
        if slow and not blocked_early:
            slow_findings, slow_errors = await self._gather(slow, text, context)
            findings.extend(slow_findings)
            errors.extend(slow_errors)
        elif slow:
            _log.debug(
                "guardrail.slow_checks_skipped",
                stage=stage.value,
                sample_id=context.sample_id,
                reason="blocked by a fast check",
                skipped=[c.rule_id for c in slow],
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        report = GuardrailReport.combine(stage, findings, latency_ms=elapsed_ms, errors=errors)

        if report.action is GuardrailAction.REDACT:
            report = report.model_copy(update={"sanitized_text": self._redact(text, findings)})

        _log.info(
            "guardrail.stage_completed",
            stage=stage.value,
            sample_id=context.sample_id,
            action=report.action.value,
            finding_count=len(report.findings),
            triggered_rules=report.triggered_rules,
            max_severity=report.max_severity.value if report.max_severity else None,
            latency_ms=round(elapsed_ms, 2),
            error_count=len(errors),
        )
        return report

    @staticmethod
    async def _gather(
        checks: Sequence[GuardrailCheck], text: str, context: CheckContext
    ) -> tuple[list[GuardrailFinding], list[str]]:
        """Run checks concurrently, collecting findings and hard failures.

        Args:
            checks: The checks to run.
            text: The text under inspection.
            context: Supporting context.

        Returns:
            A tuple of all findings and the rule ids of checks that failed
            outright. Individual check errors are normally handled inside
            :meth:`GuardrailCheck.acheck`; anything reaching here escaped that
            containment (typically cancellation).
        """
        results = await asyncio.gather(
            *(check.acheck(text, context) for check in checks),
            return_exceptions=True,
        )
        findings: list[GuardrailFinding] = []
        errors: list[str] = []
        for check, outcome in zip(checks, results, strict=True):
            if isinstance(outcome, BaseException):
                _log.error(
                    "guardrail.check_uncontained_error",
                    rule_id=check.rule_id,
                    error_type=type(outcome).__name__,
                    error=str(outcome)[:300],
                )
                errors.append(check.rule_id)
                continue
            findings.extend(outcome)
        return findings, errors

    def _redact(self, text: str, findings: Sequence[GuardrailFinding]) -> str:
        """Redact every PII span reported by the stage's findings.

        Args:
            text: The original text.
            findings: All findings from the stage.

        Returns:
            The redacted text.
        """
        entities = [entity for finding in findings for entity in finding.entities]
        return redact(text, entities, self._guardrail_settings.pii_redaction_token)

    @property
    def inbound_checks(self) -> list[GuardrailCheck]:
        """The configured inbound checks.

        Returns:
            The inbound check chain.
        """
        return list(self._inbound)

    @property
    def outbound_checks(self) -> list[GuardrailCheck]:
        """The configured outbound checks.

        Returns:
            The outbound check chain.
        """
        return list(self._outbound)

    @property
    def pii_check(self) -> PIIGuardrailCheck | None:
        """The inbound PII check, when one is configured.

        Returns:
            The check instance, or ``None``.
        """
        for check in self._inbound:
            if isinstance(check, PIIGuardrailCheck):
                return check
        return None
