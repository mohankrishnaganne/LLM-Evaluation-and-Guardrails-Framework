"""Context recall metric (retrieval coverage).

Context recall asks: *did the retriever find everything the correct answer
needs?* The ground-truth answer is decomposed into atomic statements and each
is checked for attribution to the retrieved context.

Score: ``attributable_statements / total_statements`` in [0, 1].

This is the one metric that genuinely requires a reference answer -- without it
there is no way to know what the retriever *should* have found -- so samples
lacking ``ground_truth`` are skipped rather than failed.
"""

from __future__ import annotations

from ..models import JudgeVerdict, MetricName, RAGSample
from .base import BaseEvaluator, ClaimListResponse

__all__ = ["ContextRecallEvaluator", "build_context_recall_prompt"]

_PROMPT_TEMPLATE = """\
Decompose the GROUND TRUTH answer into atomic statements, then decide for each \
whether it can be attributed to the retrieved CONTEXT.

Rules:
- An atomic statement asserts exactly one fact. Split compound sentences.
- Resolve pronouns so each statement stands alone.
- Verdicts:
  * "supported"    - the CONTEXT states or unambiguously entails the statement.
  * "contradicted" - the CONTEXT asserts something incompatible with it.
  * "unsupported"  - the CONTEXT does not cover the statement at all.
- Attribute only to the CONTEXT. Never use outside knowledge, and never credit \
a statement merely because it is true in general.
- For attributed statements, list the 1-based indices of the supporting contexts.

QUESTION:
{question}

CONTEXT:
{context}

GROUND TRUTH:
{ground_truth}

Return JSON exactly in this shape:
{{"claims": [{{"claim": "<atomic statement>", \
"verdict": "supported|contradicted|unsupported", \
"supporting_context_ids": [1], "reasoning": "<one sentence>"}}]}}
"""


def build_context_recall_prompt(question: str, context: str, ground_truth: str) -> str:
    """Render the context-recall judge prompt.

    Args:
        question: The end-user question.
        context: The enumerated retrieved context block.
        ground_truth: The reference answer to decompose.

    Returns:
        The fully-rendered prompt.
    """
    return _PROMPT_TEMPLATE.format(question=question, context=context, ground_truth=ground_truth)


class ContextRecallEvaluator(BaseEvaluator):
    """Scores how much of the reference answer the retriever actually covered."""

    metric = MetricName.CONTEXT_RECALL
    requires_ground_truth = True
    requires_contexts = True
    requires_answer = False

    async def _aevaluate(self, sample: RAGSample) -> JudgeVerdict:
        """Decompose the ground truth and check each statement's attribution.

        Args:
            sample: The sample to evaluate.

        Returns:
            A verdict whose score is the attributable-statement ratio.
        """
        prompt = build_context_recall_prompt(
            question=sample.question,
            context=self._context_block(sample),
            ground_truth=sample.ground_truth or "",
        )
        parsed, audit = await self._judge(prompt, ClaimListResponse)
        response: ClaimListResponse = parsed

        ratio = response.supported_ratio()
        claims = [judgment.to_claim() for judgment in response.claims]

        if ratio is None:
            # A ground truth that decomposes to nothing is a dataset problem,
            # not a retriever failure; score 0.0 and say so explicitly.
            return JudgeVerdict(
                score=0.0,
                reasoning=(
                    "No atomic statements could be extracted from the ground truth; "
                    "recall is undefined and reported as 0.0."
                ),
                claims=claims,
                raw={**audit, "statement_count": 0, "degenerate_ground_truth": True},
            )

        missing = [c.text for c in claims if not c.is_supported]
        reasoning = (
            f"The context covers all {len(claims)} ground-truth statement(s)."
            if not missing
            else "{n} of {total} ground-truth statement(s) were not retrieved: {examples}".format(
                n=len(missing), total=len(claims), examples="; ".join(missing[:3])
            )
        )
        return JudgeVerdict(
            score=ratio,
            reasoning=reasoning,
            claims=claims,
            raw={
                **audit,
                "statement_count": len(claims),
                "missing_count": len(missing),
            },
        )
