"""Base abstractions for inbound and outbound guardrail checks.

A *check* is one policy test over a piece of text plus optional context. Checks
are small, independent and individually testable; the
:class:`~llm_eval_guardrails.guardrails.pipeline.GuardrailPipeline` composes
them and reduces their findings to a single enforcement decision.

Checks fall into two performance classes, exposed as :attr:`GuardrailCheck.is_fast`:

* **Fast** -- regex and heuristic checks costing microseconds. Safe on the
  synchronous request path.
* **Slow** -- checks that call an LLM. Milliseconds to seconds; run these
  asynchronously and only where the latency budget allows.

The pipeline runs fast checks first and can short-circuit on a block, which
keeps the common case (a clean prompt) cheap and the abusive case fast to
reject.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import GuardrailSettings
from ..logging_config import get_logger
from ..models import (
    GuardrailAction,
    GuardrailFinding,
    GuardrailStage,
    PIIEntity,
    RetrievedDocument,
    Severity,
)

__all__ = ["CheckContext", "GuardrailCheck", "snippet"]

_log = get_logger(__name__)

#: Maximum characters of matched text quoted as evidence in a finding.
_EVIDENCE_CHARS = 120


def snippet(text: str, start: int, end: int, *, window: int = 20) -> str:
    """Extract a short, single-line excerpt around a matched span.

    Used to attach human-readable evidence to findings without echoing an
    entire prompt into logs or reports.

    Args:
        text: The full source text.
        start: Inclusive start offset of the match.
        end: Exclusive end offset of the match.
        window: Extra characters of context to include on each side.

    Returns:
        The excerpt with newlines collapsed, clipped to a bounded length and
        marked with ellipses where content was elided.
    """
    left = max(0, start - window)
    right = min(len(text), end + window)
    excerpt = " ".join(text[left:right].split())
    if len(excerpt) > _EVIDENCE_CHARS:
        excerpt = excerpt[:_EVIDENCE_CHARS] + "..."
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return prefix + excerpt + suffix


@dataclass(slots=True)
class CheckContext:
    """Everything a check may need beyond the text under inspection.

    Attributes:
        question: The originating user question, for outbound checks.
        contexts: The retrieved chunks a response must be grounded in.
        sample_id: Identifier of the sample, for log correlation.
        metadata: Arbitrary caller-supplied context (tenant, locale, route).
    """

    question: str | None = None
    contexts: list[RetrievedDocument] = field(default_factory=list)
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def joined_context(self, max_chars_per_doc: int = 4000) -> str:
        """Render the contexts as an enumerated block for judge prompts.

        Args:
            max_chars_per_doc: Per-chunk truncation budget.

        Returns:
            A 1-indexed listing of the contexts, or an explicit sentinel when
            no contexts are present.
        """
        if not self.contexts:
            return "[NO CONTEXT AVAILABLE]"
        return "\n\n".join(
            f"[{i}] {doc.truncated(max_chars_per_doc)}"
            for i, doc in enumerate(self.contexts, start=1)
        )


class GuardrailCheck(abc.ABC):
    """A single policy test producing zero or more findings.

    Subclasses implement :meth:`_acheck`. Timing and error containment are
    handled by :meth:`acheck`, because a guardrail that crashes must produce a
    decision (fail-open or fail-closed, per configuration) rather than take
    down the request it was meant to protect.

    Attributes:
        rule_id: Stable identifier recorded on every finding this check emits.
        stage: Whether the check inspects prompts or responses.
        is_fast: Whether the check is cheap enough for the synchronous path.
    """

    rule_id: str
    stage: GuardrailStage
    is_fast: bool = True

    def __init__(self, settings: GuardrailSettings) -> None:
        """Initialise the check.

        Args:
            settings: Guardrail policy settings.
        """
        self._settings = settings
        self._log = _log.bind(rule_id=self.rule_id, stage=self.stage.value)

    @abc.abstractmethod
    async def _acheck(self, text: str, context: CheckContext) -> list[GuardrailFinding]:
        """Run the policy test.

        Args:
            text: The text under inspection.
            context: Supporting context for the check.

        Returns:
            Zero or more findings. An empty list means the text passed.
        """

    async def acheck(self, text: str, context: CheckContext) -> list[GuardrailFinding]:
        """Run the check, containing failures according to policy.

        Args:
            text: The text under inspection.
            context: Supporting context for the check.

        Returns:
            The findings produced. If the check itself raises, the result
            depends on ``fail_closed``: a synthetic ``CRITICAL``/``BLOCK``
            finding when set, or an empty list (fail open) when not.
        """
        started = time.perf_counter()
        try:
            findings = await self._acheck(text, context)
        except Exception as exc:  # noqa: BLE001 - a broken check must still decide
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._log.exception(
                "guardrail.check_error",
                sample_id=context.sample_id,
                latency_ms=round(elapsed_ms, 2),
                fail_closed=self._settings.fail_closed,
            )
            if not self._settings.fail_closed:
                return []
            return [
                GuardrailFinding(
                    rule_id=f"{self.rule_id}.error",
                    stage=self.stage,
                    severity=Severity.CRITICAL,
                    action=GuardrailAction.BLOCK,
                    message=(
                        f"Guardrail check {self.rule_id} failed to execute and the policy is "
                        f"fail-closed: {exc}"
                    ),
                    metadata={"error_type": type(exc).__name__},
                )
            ]

        elapsed_ms = (time.perf_counter() - started) * 1000
        if findings:
            self._log.info(
                "guardrail.triggered",
                sample_id=context.sample_id,
                finding_count=len(findings),
                actions=[f.action.value for f in findings],
                severities=[f.severity.value for f in findings],
                latency_ms=round(elapsed_ms, 2),
            )
        return findings

    def finding(
        self,
        *,
        severity: Severity,
        action: GuardrailAction,
        message: str,
        score: float | None = None,
        entities: list[PIIEntity] | None = None,
        evidence: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        rule_id: str | None = None,
    ) -> GuardrailFinding:
        """Construct a finding pre-populated with this check's identity.

        Args:
            severity: How serious the finding is.
            action: The action this rule alone recommends.
            message: Human-readable explanation.
            score: Detector score, where applicable.
            entities: Detected PII spans, for detector-backed rules.
            evidence: Short quoted snippets supporting the finding.
            metadata: Rule-specific extras.
            rule_id: Override the rule id, for checks emitting sub-rules.

        Returns:
            The constructed finding.
        """
        return GuardrailFinding(
            rule_id=rule_id or self.rule_id,
            stage=self.stage,
            severity=severity,
            action=action,
            message=message,
            score=score,
            entities=entities or [],
            evidence=evidence or [],
            metadata=metadata or {},
        )
