"""Faithfulness (response grounding) metric.

Faithfulness asks: *is every factual claim in the answer supported by the
retrieved context?* It is the framework's primary hallucination signal and is
reference-free -- no ground-truth answer is required.

The metric decomposes the answer into atomic claims and adjudicates each one
against the numbered context, rather than scoring the answer holistically. A
single scalar judgement over a long answer hides a lone fabricated sentence;
claim-level verdicts surface it, and are also what the outbound guardrail
consumes to explain *which* statement was unsupported.

Score: ``supported_claims / total_claims`` in [0, 1]. An answer containing no
verifiable factual claims (a pure refusal, for example) scores 1.0, because
there is nothing present to be unfaithful about.
"""

from __future__ import annotations

from ..models import JudgeVerdict, MetricName, RAGSample
from .base import BaseEvaluator, ClaimListResponse

__all__ = ["FaithfulnessEvaluator", "build_faithfulness_prompt"]

_PROMPT_TEMPLATE = """\
Decompose the ANSWER into atomic factual claims, then judge each claim strictly \
against the CONTEXT.

Rules:
- An atomic claim states exactly one verifiable fact. Split compound sentences.
- Resolve pronouns and references so each claim stands alone.
- Ignore pure pleasantries, hedges, questions and meta-commentary \
("I hope this helps", "Let me know if..."). Do not emit claims for them.
- Verdicts:
  * "supported"    - the CONTEXT directly states or unambiguously entails the claim.
  * "contradicted" - the CONTEXT asserts something incompatible with the claim.
  * "unsupported"  - the CONTEXT is silent on the claim. Plausible, widely known \
or self-evidently true statements are still "unsupported" if the CONTEXT does not \
establish them.
- Judge only against the CONTEXT. Never use outside knowledge.
- For "supported" claims, list the 1-based indices of the contexts that establish them.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Return JSON exactly in this shape:
{{"claims": [{{"claim": "<atomic claim>", "verdict": "supported|contradicted|unsupported", \
"supporting_context_ids": [1], "reasoning": "<one sentence>"}}]}}

If the ANSWER contains no verifiable factual claims, return {{"claims": []}}.
"""


def build_faithfulness_prompt(question: str, context: str, answer: str) -> str:
    """Render the faithfulness judge prompt.

    Exposed separately so prompts can be unit-tested and reviewed without
    instantiating an LLM client.

    Args:
        question: The end-user question.
        context: The enumerated retrieved context block.
        answer: The generated answer under evaluation.

    Returns:
        The fully-rendered prompt.
    """
    return _PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)


class FaithfulnessEvaluator(BaseEvaluator):
    """Scores how well an answer is grounded in its retrieved context."""

    metric = MetricName.FAITHFULNESS
    requires_ground_truth = False
    requires_contexts = True
    requires_answer = True

    async def _aevaluate(self, sample: RAGSample) -> JudgeVerdict:
        """Extract and adjudicate the answer's claims.

        Args:
            sample: The sample to evaluate.

        Returns:
            A verdict whose score is the supported-claim ratio and whose
            ``claims`` list carries the per-claim breakdown.
        """
        prompt = build_faithfulness_prompt(
            question=sample.question,
            context=self._context_block(sample),
            answer=sample.answer,
        )
        parsed, audit = await self._judge(prompt, ClaimListResponse)
        response: ClaimListResponse = parsed

        ratio = response.supported_ratio()
        claims = [judgment.to_claim() for judgment in response.claims]

        if ratio is None:
            # No verifiable claims were made, so nothing can be ungrounded.
            return JudgeVerdict(
                score=1.0,
                reasoning="The answer makes no verifiable factual claims.",
                claims=claims,
                raw={**audit, "claim_count": 0},
            )

        unsupported = [c.text for c in claims if not c.is_supported]
        reasoning = (
            f"All {len(claims)} claim(s) are grounded in the retrieved context."
            if not unsupported
            else "{n} of {total} claim(s) are not grounded: {examples}".format(
                n=len(unsupported),
                total=len(claims),
                examples="; ".join(unsupported[:3]),
            )
        )
        return JudgeVerdict(
            score=ratio,
            reasoning=reasoning,
            claims=claims,
            raw={
                **audit,
                "claim_count": len(claims),
                "unsupported_count": len(unsupported),
            },
        )
