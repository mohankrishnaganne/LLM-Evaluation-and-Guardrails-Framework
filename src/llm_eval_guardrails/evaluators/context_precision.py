"""Context precision metric (retrieval relevance, rank-aware).

Context precision asks: *of the chunks the retriever returned, how many were
actually useful, and were the useful ones ranked highly?*

The score is mean average precision at K rather than a flat relevant/total
ratio, because rank position matters in RAG: generators attend most strongly to
early context, and a pipeline that buries its one relevant chunk at position 10
is materially worse than one that puts it first, even though both retrieve
exactly one relevant chunk.

    AP@K = sum_k (precision@k * rel_k) / number_of_relevant

The judge adjudicates each chunk independently for usefulness in answering the
question, using the ground truth as the reference when one is available and the
generated answer otherwise.
"""

from __future__ import annotations

from ..models import JudgeVerdict, MetricName, RAGSample
from .base import BaseEvaluator, StatementResponse

__all__ = ["ContextPrecisionEvaluator", "build_context_precision_prompt"]

_PROMPT_TEMPLATE = """\
For each numbered context, decide whether it contains information that is \
genuinely useful for answering the QUESTION, judged against the REFERENCE.

Rules:
- Answer with one boolean per context, in the same order, with the same count.
- true  - the context supplies at least one fact needed to produce the REFERENCE.
- false - the context is off-topic, redundant background, or merely shares \
keywords with the question without contributing a needed fact.
- Judge each context on its own. Do not credit a context for information that \
only appears in another one.

QUESTION:
{question}

REFERENCE ({reference_kind}):
{reference}

CONTEXTS ({count} total):
{context}

Return JSON exactly in this shape, with exactly {count} booleans:
{{"statements": [true, false], "reasoning": "<one sentence>"}}
"""


def build_context_precision_prompt(
    question: str, reference: str, reference_kind: str, context: str, count: int
) -> str:
    """Render the context-precision judge prompt.

    Args:
        question: The end-user question.
        reference: The reference text each chunk is judged against.
        reference_kind: Either ``"ground truth"`` or ``"generated answer"``.
        context: The enumerated retrieved context block.
        count: Number of retrieved chunks; the judge must return this many booleans.

    Returns:
        The fully-rendered prompt.
    """
    return _PROMPT_TEMPLATE.format(
        question=question,
        reference=reference,
        reference_kind=reference_kind,
        context=context,
        count=count,
    )


def average_precision(relevance: list[bool]) -> float:
    """Compute mean average precision at K over a ranked relevance list.

    Args:
        relevance: Per-position relevance flags, in retrieval order.

    Returns:
        AP@K in [0, 1]; ``0.0`` when nothing is relevant.
    """
    hits = 0
    precision_sum = 0.0
    for index, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / index
    if hits == 0:
        return 0.0
    return precision_sum / hits


class ContextPrecisionEvaluator(BaseEvaluator):
    """Scores retrieval relevance with rank-aware average precision."""

    metric = MetricName.CONTEXT_PRECISION
    requires_ground_truth = False
    requires_contexts = True
    requires_answer = False

    async def _aevaluate(self, sample: RAGSample) -> JudgeVerdict:
        """Judge each retrieved chunk and compute AP@K.

        Args:
            sample: The sample to evaluate.

        Returns:
            A verdict whose score is AP@K over the judged relevance flags.
        """
        reference = (sample.ground_truth or "").strip()
        reference_kind = "ground truth"
        if not reference:
            reference = sample.answer.strip() or "[no answer available]"
            reference_kind = "generated answer"

        count = len(sample.contexts)
        prompt = build_context_precision_prompt(
            question=sample.question,
            reference=reference,
            reference_kind=reference_kind,
            context=self._context_block(sample),
            count=count,
        )
        parsed, audit = await self._judge(prompt, StatementResponse)
        response: StatementResponse = parsed

        # Judges occasionally return the wrong number of verdicts. Truncating
        # or padding with False is the conservative repair: it never inflates
        # the score, and it keeps one malformed reply from failing the sample.
        verdicts = list(response.statements[:count])
        truncated = len(response.statements) > count
        padded = count - len(verdicts)
        verdicts.extend([False] * padded)

        score = average_precision(verdicts)
        relevant = sum(1 for v in verdicts if v)
        return JudgeVerdict(
            score=score,
            reasoning=response.reasoning
            or f"{relevant} of {count} retrieved chunk(s) were judged useful.",
            raw={
                **audit,
                "retrieved_count": count,
                "relevant_count": relevant,
                "relevance": verdicts,
                "judge_returned_wrong_count": bool(truncated or padded),
            },
        )
