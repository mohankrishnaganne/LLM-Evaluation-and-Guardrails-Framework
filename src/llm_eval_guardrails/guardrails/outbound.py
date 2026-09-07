"""Outbound (response-side) guardrail checks.

These checks run after the RAG pipeline has generated an answer but before it
reaches the user. Two classes of risk are covered:

* **Unsupported claims** -- the answer asserts facts the retrieved context does
  not establish. This is the hallucination guardrail, and it is the one check
  that requires an LLM call, so it is marked slow.
* **Leakage and policy** -- the answer echoes PII or credentials, or matches a
  prohibited phrase. These are fast regex checks.

The grounding check reuses the faithfulness judge rather than a separate
prompt. That is deliberate: the guardrail and the offline metric should agree
about what "grounded" means, otherwise a system can pass its evaluation suite
and still be blocked in production, or vice versa.
"""

from __future__ import annotations

from typing import Final

from ..config import GuardrailSettings, Settings
from ..evaluators.base import JUDGE_SYSTEM_PROMPT, ClaimListResponse
from ..evaluators.faithfulness import build_faithfulness_prompt
from ..llm.base import ChatMessage, LLMClient
from ..models import (
    ClaimVerdict,
    GuardrailAction,
    GuardrailFinding,
    GuardrailStage,
    Severity,
)
from .base import CheckContext, GuardrailCheck, snippet
from .inbound import BlockedTopicCheck
from .pii import PIIDetector

__all__ = [
    "ResponseLeakageCheck",
    "UnsupportedClaimCheck",
    "build_outbound_checks",
]

#: Entity types whose presence in a *response* is always a security incident.
_SECRET_TYPES: Final[frozenset[str]] = frozenset(
    {"aws_access_key", "aws_secret_key", "private_key", "jwt"}
)


class UnsupportedClaimCheck(GuardrailCheck):
    """Blocks or flags responses whose claims are not grounded in the context.

    The judge decomposes the response into atomic claims and adjudicates each
    against the retrieved context. Two conditions are treated differently:

    * A **contradicted** claim always blocks. The context actively disagrees
      with the answer, which is the most dangerous failure mode in a RAG
      system and is never acceptable to serve.
    * A **low supported ratio** blocks only when ``block_unsupported_claims``
      is set; otherwise it flags. Teams typically roll this out in flag mode
      first to calibrate ``grounding_threshold`` against real traffic before
      turning on enforcement.

    This check issues one LLM call and is therefore *not* suitable for a
    latency budget measured in single-digit milliseconds.
    """

    rule_id = "outbound.unsupported_claims"
    stage = GuardrailStage.OUTBOUND
    is_fast = False

    def __init__(
        self,
        settings: GuardrailSettings,
        llm: LLMClient,
        *,
        max_context_chars: int = 4000,
    ) -> None:
        """Initialise the check.

        Args:
            settings: Guardrail policy settings.
            llm: The judge LLM used to adjudicate claims.
            max_context_chars: Per-chunk truncation budget for the judge prompt.
        """
        super().__init__(settings)
        self._llm = llm
        self._max_context_chars = max_context_chars

    async def _acheck(self, text: str, context: CheckContext) -> list[GuardrailFinding]:
        """Adjudicate the response's claims against the retrieved context.

        Args:
            text: The generated response.
            context: Must carry the retrieved contexts and, ideally, the question.

        Returns:
            Findings describing contradicted and unsupported claims.

        Raises:
            FrameworkError: If the judge call fails. The base class converts
                this into a fail-open or fail-closed decision per policy, which
                is the correct place for that judgement -- a guardrail that
                cannot reach its judge is a policy question, not a bug here.
        """
        if not text.strip():
            return []

        if not context.contexts:
            # With no retrieved context, nothing in the answer can be grounded.
            return [
                self.finding(
                    rule_id="outbound.no_context",
                    severity=Severity.HIGH,
                    action=(
                        GuardrailAction.BLOCK
                        if self._settings.block_unsupported_claims
                        else GuardrailAction.FLAG
                    ),
                    message=(
                        "Response was produced with no retrieved context, so none of "
                        "its claims can be verified as grounded."
                    ),
                    score=0.0,
                )
            ]

        prompt = build_faithfulness_prompt(
            question=context.question or "[question unavailable]",
            context=context.joined_context(self._max_context_chars),
            answer=text,
        )
        parsed, _response = await self._llm.acomplete_json(
            [ChatMessage.user(prompt)],
            schema=ClaimListResponse,
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
        )
        judged: ClaimListResponse = parsed

        if not judged.claims:
            # No verifiable claims means nothing to be ungrounded about.
            return []

        contradicted = [c for c in judged.claims if c.verdict is ClaimVerdict.CONTRADICTED]
        unsupported = [c for c in judged.claims if c.verdict is ClaimVerdict.UNSUPPORTED]
        ratio = judged.supported_ratio() or 0.0

        findings: list[GuardrailFinding] = []

        if contradicted:
            findings.append(
                self.finding(
                    rule_id="outbound.contradicted_claim",
                    severity=Severity.CRITICAL,
                    action=GuardrailAction.BLOCK,
                    message=(
                        f"Response contains {len(contradicted)} claim(s) that the "
                        "retrieved context contradicts."
                    ),
                    score=ratio,
                    evidence=[c.claim for c in contradicted[:3]],
                    metadata={
                        "contradicted_claims": [c.claim for c in contradicted],
                        "supported_ratio": round(ratio, 4),
                    },
                )
            )

        if ratio < self._settings.grounding_threshold:
            block = self._settings.block_unsupported_claims
            grounded = len(judged.claims) - len(contradicted) - len(unsupported)
            findings.append(
                self.finding(
                    severity=Severity.HIGH if block else Severity.MEDIUM,
                    action=GuardrailAction.BLOCK if block else GuardrailAction.FLAG,
                    message=(
                        f"Only {grounded} of {len(judged.claims)} claim(s) are grounded "
                        f"(ratio {ratio:.2f} < threshold "
                        f"{self._settings.grounding_threshold:.2f})."
                    ),
                    score=ratio,
                    evidence=[c.claim for c in unsupported[:3]],
                    metadata={
                        "supported_ratio": round(ratio, 4),
                        "threshold": self._settings.grounding_threshold,
                        "claim_count": len(judged.claims),
                        "unsupported_claims": [c.claim for c in unsupported],
                    },
                )
            )

        return findings


class ResponseLeakageCheck(GuardrailCheck):
    """Blocks responses that leak credentials or personal data.

    A response echoing PII is a different risk from a prompt containing it: the
    prompt's PII came from the user, who already knows it, whereas a response's
    PII may have been retrieved from a document the requester should never have
    seen. Personal data therefore blocks by default here, and credentials
    always block.
    """

    rule_id = "outbound.pii_leak"
    stage = GuardrailStage.OUTBOUND
    is_fast = True

    def __init__(self, settings: GuardrailSettings, *, detector: PIIDetector | None = None) -> None:
        """Initialise the check.

        Args:
            settings: Guardrail policy settings.
            detector: Pre-built detector; one is created from settings when
                omitted.
        """
        super().__init__(settings)
        self._detector = detector or PIIDetector(
            allowed_types=settings.pii_entities,
            use_presidio=settings.use_presidio,
        )

    async def _acheck(self, text: str, context: CheckContext) -> list[GuardrailFinding]:
        """Scan the response for PII and credentials.

        Personal data already present in the originating question is not
        reported: repeating back what the user just supplied is not a leak, and
        flagging it would bury genuine findings in noise.

        Args:
            text: The generated response.
            context: Supplies the originating question, when available.

        Returns:
            Findings for leaked credentials and personal data.
        """
        entities = self._detector.detect(text)
        if not entities:
            return []

        question = (context.question or "").lower()
        findings: list[GuardrailFinding] = []

        secrets = [e for e in entities if e.entity_type.value in _SECRET_TYPES]
        if secrets:
            findings.append(
                self.finding(
                    rule_id="outbound.secret_leak",
                    severity=Severity.CRITICAL,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Response contains {count} credential-like value(s) ({types})."
                    ).format(
                        count=len(secrets),
                        types=", ".join(sorted({e.entity_type.value for e in secrets})),
                    ),
                    entities=secrets,
                    metadata={"entity_count": len(secrets)},
                )
            )

        personal = [
            e
            for e in entities
            if e.entity_type.value not in _SECRET_TYPES
            and text[e.start : e.end].lower() not in question
        ]
        if personal:
            findings.append(
                self.finding(
                    severity=Severity.HIGH,
                    action=GuardrailAction.BLOCK
                    if self._settings.block_on_pii
                    else GuardrailAction.REDACT,
                    message=(
                        "Response contains {count} PII entity/entities ({types}) that "
                        "were not present in the question."
                    ).format(
                        count=len(personal),
                        types=", ".join(sorted({e.entity_type.value for e in personal})),
                    ),
                    entities=personal,
                    evidence=[snippet(text, e.start, e.end) for e in personal[:3]],
                    metadata={"entity_count": len(personal)},
                )
            )

        return findings


def build_outbound_checks(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    detector: PIIDetector | None = None,
) -> list[GuardrailCheck]:
    """Assemble the outbound check chain from configuration.

    The grounding check is included only when an LLM is available and
    ``use_llm_for_outbound`` is enabled, so a deployment can run the fast
    leakage and policy checks alone when its latency budget forbids a judge
    call on the response path.

    Args:
        settings: The full application settings.
        llm: The judge LLM, required for the grounding check.
        detector: Shared PII detector, to avoid re-initialising Presidio.

    Returns:
        The checks to run against outbound responses, cheapest first.
    """
    guardrail_settings = settings.guardrails
    checks: list[GuardrailCheck] = [ResponseLeakageCheck(guardrail_settings, detector=detector)]
    if guardrail_settings.blocked_topics:
        checks.append(BlockedTopicCheck(guardrail_settings, stage=GuardrailStage.OUTBOUND))
    if llm is not None and guardrail_settings.use_llm_for_outbound:
        checks.append(
            UnsupportedClaimCheck(
                guardrail_settings,
                llm,
                max_context_chars=settings.evaluation.max_context_chars,
            )
        )
    return checks
