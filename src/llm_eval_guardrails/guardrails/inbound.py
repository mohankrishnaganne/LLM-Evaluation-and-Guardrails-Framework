"""Inbound (prompt-side) guardrail checks.

Every check here is regex- or heuristic-based and therefore fast enough to run
synchronously on the request path before a prompt reaches the RAG pipeline. No
model is loaded and no network call is made, so the added latency is
microseconds and independent of provider availability.

Checks provided:

* :class:`PromptLengthCheck` -- rejects oversized prompts (a cost and
  context-window control, and a cheap denial-of-wallet defence).
* :class:`PIIGuardrailCheck` -- detects PII and secrets, redacting or blocking
  according to policy.
* :class:`PromptInjectionCheck` -- scores injection and jailbreak heuristics.
* :class:`BlockedTopicCheck` -- matches configured prohibited phrases.
"""

from __future__ import annotations

import re
from typing import Final

from ..config import GuardrailSettings
from ..models import GuardrailAction, GuardrailFinding, GuardrailStage, Severity
from .base import CheckContext, GuardrailCheck, snippet
from .injection import INJECTION_SIGNALS, scan_for_injection
from .pii import PIIDetector, redact

__all__ = [
    "BlockedTopicCheck",
    "PIIGuardrailCheck",
    "PromptInjectionCheck",
    "PromptLengthCheck",
    "build_inbound_checks",
]

#: Entity types treated as credentials rather than personal data. Leaking these
#: is a security incident, so they always block regardless of the PII policy.
_SECRET_TYPES: Final[frozenset[str]] = frozenset(
    {"aws_access_key", "aws_secret_key", "private_key", "jwt"}
)


class PromptLengthCheck(GuardrailCheck):
    """Rejects prompts exceeding the configured character budget."""

    rule_id = "inbound.prompt_length"
    stage = GuardrailStage.INBOUND
    is_fast = True

    async def _acheck(self, text: str, context: CheckContext) -> list[GuardrailFinding]:
        """Compare the prompt length against the configured maximum.

        Args:
            text: The prompt under inspection.
            context: Unused supporting context.

        Returns:
            One blocking finding when the prompt is too long, else nothing.
        """
        limit = self._settings.max_prompt_chars
        length = len(text)
        if length <= limit:
            return []
        return [
            self.finding(
                severity=Severity.MEDIUM,
                action=GuardrailAction.BLOCK,
                message=(f"Prompt is {length} characters, exceeding the {limit}-character limit."),
                metadata={"length": length, "limit": limit},
            )
        ]


class PIIGuardrailCheck(GuardrailCheck):
    """Detects PII and secrets, redacting or blocking per policy.

    Credentials (API keys, private keys, JWTs) always block: unlike a personal
    email address, a leaked key cannot be made safe by redacting it from the
    prompt, because it has already been transmitted and must be treated as
    compromised. Personal data blocks or redacts according to ``block_on_pii``.
    """

    rule_id = "inbound.pii"
    stage = GuardrailStage.INBOUND
    is_fast = True

    def __init__(self, settings: GuardrailSettings, *, detector: PIIDetector | None = None) -> None:
        """Initialise the check.

        Args:
            settings: Guardrail policy settings.
            detector: Pre-built detector; one is created from settings when
                omitted. Injecting a detector keeps tests deterministic and
                avoids repeated Presidio initialisation.
        """
        super().__init__(settings)
        self._detector = detector or PIIDetector(
            allowed_types=settings.pii_entities,
            use_presidio=settings.use_presidio,
        )

    async def _acheck(self, text: str, context: CheckContext) -> list[GuardrailFinding]:
        """Scan the prompt for PII and secrets.

        Args:
            text: The prompt under inspection.
            context: Unused supporting context.

        Returns:
            At most two findings: one for secrets (blocking) and one for
            personal data (blocking or redacting).
        """
        entities = self._detector.detect(text)
        if not entities:
            return []

        secrets = [e for e in entities if e.entity_type.value in _SECRET_TYPES]
        personal = [e for e in entities if e.entity_type.value not in _SECRET_TYPES]
        findings: list[GuardrailFinding] = []

        if secrets:
            findings.append(
                GuardrailFinding(
                    rule_id="inbound.secret",
                    stage=self.stage,
                    severity=Severity.CRITICAL,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Prompt contains {count} credential-like value(s) ({types}). "
                        "Treat the exposed credentials as compromised and rotate them."
                    ).format(
                        count=len(secrets),
                        types=", ".join(sorted({e.entity_type.value for e in secrets})),
                    ),
                    entities=secrets,
                    # Secret values are never quoted as evidence.
                    metadata={"entity_count": len(secrets)},
                )
            )

        if personal:
            block = self._settings.block_on_pii
            findings.append(
                GuardrailFinding(
                    rule_id=self.rule_id,
                    stage=self.stage,
                    severity=Severity.HIGH if block else Severity.MEDIUM,
                    action=GuardrailAction.BLOCK if block else GuardrailAction.REDACT,
                    message=(
                        "Prompt contains {count} PII entity/entities ({types}); action: {action}."
                    ).format(
                        count=len(personal),
                        types=", ".join(sorted({e.entity_type.value for e in personal})),
                        action="blocked" if block else "redacted",
                    ),
                    entities=personal,
                    evidence=[snippet(text, e.start, e.end) for e in personal[:3]],
                    metadata={
                        "entity_count": len(personal),
                        "detectors": sorted({e.detector for e in personal}),
                    },
                )
            )

        return findings

    def redact_text(self, text: str, findings: list[GuardrailFinding]) -> str:
        """Apply redaction for every PII entity carried by ``findings``.

        Args:
            text: The original prompt.
            findings: Findings whose entities should be redacted.

        Returns:
            The redacted prompt.
        """
        entities = [entity for finding in findings for entity in finding.entities]
        return redact(text, entities, self._settings.pii_redaction_token)


class PromptInjectionCheck(GuardrailCheck):
    """Scores a prompt against injection and jailbreak heuristics.

    Scores at or above ``injection_threshold`` block; anything lower that still
    matched at least one signal is flagged for review rather than dropped,
    because a single weak heuristic is not enough evidence to refuse a user's
    request outright.
    """

    rule_id = "inbound.prompt_injection"
    stage = GuardrailStage.INBOUND
    is_fast = True

    async def _acheck(self, text: str, context: CheckContext) -> list[GuardrailFinding]:
        """Run the injection heuristics.

        Args:
            text: The prompt under inspection.
            context: Unused supporting context.

        Returns:
            One finding when any signal matched, else nothing.
        """
        scan = scan_for_injection(text)
        if scan.clean:
            return []

        blocking = scan.score >= self._settings.injection_threshold
        descriptions = {signal.signal_id: signal.description for signal in INJECTION_SIGNALS}
        return [
            self.finding(
                severity=Severity.HIGH if blocking else Severity.LOW,
                action=GuardrailAction.BLOCK if blocking else GuardrailAction.FLAG,
                message=(
                    "Prompt matched {count} injection heuristic(s) "
                    "(score {score:.2f}, threshold {threshold:.2f}): {signals}"
                ).format(
                    count=len(scan.signals),
                    score=scan.score,
                    threshold=self._settings.injection_threshold,
                    signals="; ".join(descriptions.get(s, s) for s in scan.signals[:3]),
                ),
                score=scan.score,
                evidence=list(scan.evidence[:3]),
                metadata={
                    "signals": list(scan.signals),
                    "threshold": self._settings.injection_threshold,
                },
            )
        ]


class BlockedTopicCheck(GuardrailCheck):
    """Blocks text containing any configured prohibited phrase.

    Phrases are matched case-insensitively on word boundaries, so ``"class"``
    does not match ``"classification"``. This is a deliberately blunt policy
    hook for organisation-specific prohibitions; nuanced topic moderation
    belongs in a dedicated classifier.
    """

    rule_id = "blocked_topic"
    is_fast = True

    def __init__(
        self, settings: GuardrailSettings, *, stage: GuardrailStage = GuardrailStage.INBOUND
    ) -> None:
        """Initialise the check and pre-compile the phrase patterns.

        Args:
            settings: Guardrail policy settings supplying ``blocked_topics``.
            stage: The stage this instance guards; the same rule is reused for
                inbound prompts and outbound responses.
        """
        self.stage = stage
        super().__init__(settings)
        self._patterns = [
            (
                phrase,
                re.compile(
                    rf"(?<!\w){re.escape(phrase.strip())}(?!\w)",
                    re.IGNORECASE,
                ),
            )
            for phrase in settings.blocked_topics
            if phrase.strip()
        ]

    async def _acheck(self, text: str, context: CheckContext) -> list[GuardrailFinding]:
        """Match the text against every configured prohibited phrase.

        Args:
            text: The text under inspection.
            context: Unused supporting context.

        Returns:
            One finding listing all matched phrases, else nothing.
        """
        if not self._patterns:
            return []

        matched: list[str] = []
        evidence: list[str] = []
        for phrase, pattern in self._patterns:
            match = pattern.search(text)
            if match is not None:
                matched.append(phrase)
                if len(evidence) < 3:
                    evidence.append(snippet(text, match.start(), match.end()))

        if not matched:
            return []

        return [
            self.finding(
                rule_id=f"{self.stage.value}.blocked_topic",
                severity=Severity.HIGH,
                action=GuardrailAction.BLOCK,
                message="Text matched {count} blocked topic phrase(s): {phrases}".format(
                    count=len(matched), phrases=", ".join(matched[:5])
                ),
                evidence=evidence,
                metadata={"matched_phrases": matched},
            )
        ]


def build_inbound_checks(
    settings: GuardrailSettings, *, detector: PIIDetector | None = None
) -> list[GuardrailCheck]:
    """Assemble the inbound check chain from configuration.

    Args:
        settings: Guardrail policy settings.
        detector: Shared PII detector, to avoid re-initialising Presidio.

    Returns:
        The checks to run against inbound prompts, cheapest first.
    """
    checks: list[GuardrailCheck] = [
        PromptLengthCheck(settings),
        PromptInjectionCheck(settings),
        PIIGuardrailCheck(settings, detector=detector),
    ]
    if settings.blocked_topics:
        checks.append(BlockedTopicCheck(settings, stage=GuardrailStage.INBOUND))
    return checks
