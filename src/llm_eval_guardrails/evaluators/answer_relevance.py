"""Answer relevance metric (answer quality).

Answer relevance asks: *does the answer actually address the question that was
asked?* It is orthogonal to faithfulness -- an answer can be perfectly grounded
in the retrieved context while answering a different question, or hedging so
heavily that it conveys nothing.

The judge scores three sub-dimensions and the metric combines them:

* **directness** -- does it address the specific question asked?
* **completeness** -- does it cover the question's scope, not just part of it?
* **conciseness** -- is it free of padding and irrelevant digression?

An answer flagged as non-committal ("I don't know", "the context does not say")
is scored 0.0 regardless of the sub-scores. A refusal may be the *correct*
behaviour for an ungrounded question, but it is not a relevant answer, and
conflating the two would let a system that refuses everything post a healthy
relevance score. The distinction is preserved in ``metadata.noncommittal`` so
reports can separate honest refusals from genuine irrelevance.
"""

from __future__ import annotations

from pydantic import Field

from ..models import JudgeVerdict, MetricName, RAGSample
from .base import BaseEvaluator, JudgeSchema

__all__ = ["AnswerRelevanceEvaluator", "AnswerRelevanceResponse", "build_answer_relevance_prompt"]

_WEIGHTS: dict[str, float] = {"directness": 0.5, "completeness": 0.3, "conciseness": 0.2}

_PROMPT_TEMPLATE = """\
Judge how well the ANSWER responds to the QUESTION.

Score each dimension from 0.0 to 1.0:
- directness:   Does it address the specific question asked, rather than an \
adjacent or broader one?
- completeness: Does it cover the full scope of the question, including every \
part of a multi-part question?
- conciseness:  Is it free of padding, repetition and irrelevant digression? \
A brief answer that fully responds scores 1.0.

Also set "noncommittal" to true if the ANSWER declines to answer, says it does \
not know, or states that the information is unavailable. An answer that gives \
substantive information and merely adds a caveat is NOT noncommittal.

Judge only the fit between QUESTION and ANSWER. Do not reward or penalise \
factual accuracy here; that is measured separately.

QUESTION:
{question}

ANSWER:
{answer}

Return JSON exactly in this shape:
{{"directness": 0.0, "completeness": 0.0, "conciseness": 0.0, \
"noncommittal": false, "reasoning": "<one or two sentences>"}}
"""


class AnswerRelevanceResponse(JudgeSchema):
    """Judge response for the answer-relevance metric."""

    directness: float = Field(ge=0.0, le=1.0, description="Addresses the specific question.")
    completeness: float = Field(ge=0.0, le=1.0, description="Covers the question's full scope.")
    conciseness: float = Field(ge=0.0, le=1.0, description="Free of padding and digression.")
    noncommittal: bool = Field(default=False, description="Whether the answer declines to answer.")
    reasoning: str | None = Field(default=None, description="Brief justification.")

    def weighted_score(self) -> float:
        """Combine the sub-scores using the metric's fixed weights.

        Returns:
            The weighted score in [0, 1].
        """
        return (
            self.directness * _WEIGHTS["directness"]
            + self.completeness * _WEIGHTS["completeness"]
            + self.conciseness * _WEIGHTS["conciseness"]
        )


def build_answer_relevance_prompt(question: str, answer: str) -> str:
    """Render the answer-relevance judge prompt.

    Args:
        question: The end-user question.
        answer: The generated answer under evaluation.

    Returns:
        The fully-rendered prompt.
    """
    return _PROMPT_TEMPLATE.format(question=question, answer=answer)


class AnswerRelevanceEvaluator(BaseEvaluator):
    """Scores how directly and completely an answer addresses its question."""

    metric = MetricName.ANSWER_RELEVANCE
    requires_ground_truth = False
    requires_contexts = False
    requires_answer = True

    async def _aevaluate(self, sample: RAGSample) -> JudgeVerdict:
        """Score the question/answer fit.

        Args:
            sample: The sample to evaluate.

        Returns:
            A verdict whose score is the weighted sub-score combination, or
            ``0.0`` when the answer is non-committal.
        """
        prompt = build_answer_relevance_prompt(sample.question, sample.answer)
        parsed, audit = await self._judge(prompt, AnswerRelevanceResponse)
        response: AnswerRelevanceResponse = parsed

        weighted = response.weighted_score()
        score = 0.0 if response.noncommittal else weighted
        reasoning = response.reasoning
        if response.noncommittal:
            reasoning = "Answer is non-committal (scored 0.0). {detail}".format(
                detail=reasoning or ""
            ).strip()

        return JudgeVerdict(
            score=score,
            reasoning=reasoning,
            raw={
                **audit,
                "directness": response.directness,
                "completeness": response.completeness,
                "conciseness": response.conciseness,
                "noncommittal": response.noncommittal,
                "weighted_before_penalty": round(weighted, 4),
            },
        )
