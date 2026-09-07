"""Dataset loading and normalisation.

Evaluation datasets arrive from many places -- exported traces, spreadsheets,
hand-curated golden sets -- and rarely agree on field names. This module
normalises the common variants into :class:`~llm_eval_guardrails.models.RAGSample`
and validates the result, so downstream code can assume a single shape.

Supported inputs:

* **JSONL** -- one object per line. Contexts may be a list of strings, a list
  of objects, or a JSON-encoded string.
* **CSV** -- one row per sample. Contexts are split on a delimiter or parsed as
  embedded JSON, since CSV has no native list type.

Malformed rows are reported with their index and, by default, skipped rather
than failing the run: a single bad row in a 10,000-row export should not cost a
nightly evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from pydantic import ValidationError

from ..config import AWSSettings
from ..exceptions import DatasetError
from ..logging_config import get_logger
from ..models import RAGSample, RetrievedDocument
from .s3 import S3Storage

__all__ = ["DatasetLoader", "normalize_record"]

_log = get_logger(__name__)

#: Accepted aliases for each canonical field, in priority order.
_FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "sample_id": ("sample_id", "id", "sampleId", "question_id", "trace_id", "uuid"),
    "question": ("question", "query", "input", "prompt", "user_input"),
    "answer": ("answer", "response", "output", "generated_answer", "prediction"),
    "contexts": (
        "contexts",
        "context",
        "retrieved_contexts",
        "retrieved_documents",
        "documents",
        "chunks",
        "source_documents",
    ),
    "ground_truth": (
        "ground_truth",
        "ground_truths",
        "reference",
        "expected_answer",
        "gold_answer",
        "target",
    ),
    "metadata": ("metadata", "meta", "tags", "attributes"),
}

#: Delimiters tried, in order, when splitting CSV context cells.
_CSV_CONTEXT_DELIMITERS: Final[tuple[str, ...]] = ("|||", "\n---\n", "||")


def _first_present(record: Mapping[str, Any], field: str) -> Any:
    """Return the first aliased value present for a canonical field.

    Args:
        record: The raw dataset record.
        field: The canonical field name.

    Returns:
        The value found, or ``None`` when no alias is present.
    """
    for alias in _FIELD_ALIASES[field]:
        if alias in record and record[alias] is not None:
            return record[alias]
    return None


def _coerce_contexts(raw: Any) -> list[RetrievedDocument]:
    """Normalise a record's contexts into retrieved documents.

    Accepts a list of strings, a list of objects with a text-like field, a
    single string, or a JSON-encoded representation of any of those.

    Args:
        raw: The raw contexts value.

    Returns:
        The normalised documents, skipping entries with no usable text.
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        # A JSON-encoded list is common when contexts survive a CSV round trip.
        if stripped[0] in "[{":
            try:
                return _coerce_contexts(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        for delimiter in _CSV_CONTEXT_DELIMITERS:
            if delimiter in stripped:
                return _coerce_contexts(
                    [part for part in stripped.split(delimiter) if part.strip()]
                )
        return _coerce_contexts([stripped])

    if isinstance(raw, Mapping):
        return _coerce_contexts([raw])

    if not isinstance(raw, Sequence):
        return []

    documents: list[RetrievedDocument] = []
    for index, item in enumerate(raw):
        document = _coerce_one_context(item, index)
        if document is not None:
            documents.append(document)
    return documents


def _coerce_one_context(item: Any, index: int) -> RetrievedDocument | None:
    """Normalise a single context entry.

    Args:
        item: The raw entry, either a string or a mapping.
        index: Zero-based position, used for the fallback document id and rank.

    Returns:
        The document, or ``None`` when the entry has no usable text.
    """
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return RetrievedDocument(doc_id=f"ctx-{index}", content=text, rank=index)

    if isinstance(item, Mapping):
        text = ""
        for key in ("content", "text", "page_content", "chunk", "body", "passage"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        if not text:
            return None

        doc_id = item.get("doc_id") or item.get("id") or f"ctx-{index}"
        score = item.get("score")
        metadata = item.get("metadata")
        return RetrievedDocument(
            doc_id=str(doc_id),
            content=text,
            score=float(score) if isinstance(score, (int, float)) else None,
            rank=index,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    return None


def _coerce_ground_truth(raw: Any) -> str | None:
    """Normalise a ground-truth value that may be a string or a list.

    Datasets exported from some frameworks store ``ground_truths`` as a list of
    acceptable answers; the first non-empty entry is used.

    Args:
        raw: The raw ground-truth value.

    Returns:
        The reference answer, or ``None`` when absent or empty.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw.strip() or None
    if isinstance(raw, Sequence):
        for item in raw:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _coerce_metadata(raw: Any) -> dict[str, Any]:
    """Normalise a record's metadata into a dictionary.

    Args:
        raw: The raw metadata value; a mapping or a JSON-encoded object.

    Returns:
        The metadata dictionary, empty when absent or unparseable.
    """
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def normalize_record(record: Mapping[str, Any], index: int) -> RAGSample:
    """Normalise one raw dataset record into a validated sample.

    Args:
        record: The raw record.
        index: Zero-based position in the dataset, used to synthesise a
            ``sample_id`` when the record has none.

    Returns:
        The validated sample.

    Raises:
        DatasetError: If the record lacks a question or otherwise fails
            validation.
    """
    question = _first_present(record, "question")
    if not isinstance(question, str) or not question.strip():
        msg = "record {index} has no usable question field (looked for: {aliases})".format(
            index=index, aliases=", ".join(_FIELD_ALIASES["question"])
        )
        raise DatasetError(msg)

    answer = _first_present(record, "answer")
    sample_id = _first_present(record, "sample_id")

    try:
        return RAGSample(
            sample_id=str(sample_id) if sample_id is not None else f"sample-{index}",
            question=question.strip(),
            answer=answer.strip() if isinstance(answer, str) else "",
            contexts=_coerce_contexts(_first_present(record, "contexts")),
            ground_truth=_coerce_ground_truth(_first_present(record, "ground_truth")),
            metadata=_coerce_metadata(_first_present(record, "metadata")),
        )
    except ValidationError as exc:
        msg = f"record {index} failed validation: {exc}"
        raise DatasetError(msg) from exc


class DatasetLoader:
    """Loads evaluation datasets from S3 or the local filesystem.

    Example:
        >>> loader = DatasetLoader(aws_settings)  # doctest: +SKIP
        >>> samples = await loader.aload("s3://bucket/eval/golden.jsonl")  # doctest: +SKIP
    """

    def __init__(self, settings: AWSSettings, *, storage: S3Storage | None = None) -> None:
        """Initialise the loader.

        Args:
            settings: AWS settings used to build the storage client.
            storage: Pre-built storage wrapper, primarily for tests.
        """
        self._storage = storage or S3Storage(settings)

    @property
    def storage(self) -> S3Storage:
        """The underlying storage wrapper.

        Returns:
            The storage instance.
        """
        return self._storage

    def load(self, uri: str, *, strict: bool = False, limit: int | None = None) -> list[RAGSample]:
        """Load and normalise a dataset.

        The format is inferred from the URI suffix: ``.csv`` is read as CSV and
        everything else as JSONL.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.
            strict: Raise on the first malformed record instead of skipping it.
            limit: Stop after this many *valid* samples; useful for smoke runs.

        Returns:
            The validated samples.

        Raises:
            DatasetError: If the dataset is empty, or if ``strict`` is set and
                any record is malformed.
        """
        records = (
            self._storage.read_csv(uri)
            if uri.lower().endswith(".csv")
            else self._storage.read_jsonl(uri)
        )
        samples = self._normalize_all(records, uri=uri, strict=strict, limit=limit)

        if not samples:
            msg = f"dataset {uri} produced no valid samples"
            raise DatasetError(msg)

        _log.info(
            "dataset.loaded",
            uri=uri,
            record_count=len(records),
            sample_count=len(samples),
            with_ground_truth=sum(1 for s in samples if s.ground_truth),
        )
        return samples

    async def aload(
        self, uri: str, *, strict: bool = False, limit: int | None = None
    ) -> list[RAGSample]:
        """Async wrapper over :meth:`load`.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.
            strict: Raise on the first malformed record instead of skipping it.
            limit: Stop after this many valid samples.

        Returns:
            The validated samples.
        """
        import asyncio

        return await asyncio.to_thread(self.load, uri, strict=strict, limit=limit)

    @staticmethod
    def _normalize_all(
        records: Iterable[Mapping[str, Any]],
        *,
        uri: str,
        strict: bool,
        limit: int | None,
    ) -> list[RAGSample]:
        """Normalise every record, honouring the strictness and limit policy.

        Args:
            records: The raw records.
            uri: Source URI, used in log events.
            strict: Raise on the first malformed record.
            limit: Stop after this many valid samples.

        Returns:
            The validated samples.

        Raises:
            DatasetError: If ``strict`` is set and a record is malformed.
        """
        samples: list[RAGSample] = []
        skipped = 0

        for index, record in enumerate(records):
            if limit is not None and len(samples) >= limit:
                break
            try:
                samples.append(normalize_record(record, index))
            except DatasetError as exc:
                if strict:
                    raise
                skipped += 1
                _log.warning("dataset.record_skipped", uri=uri, index=index, error=str(exc)[:300])

        if skipped:
            _log.warning("dataset.records_skipped", uri=uri, skipped=skipped)
        return samples
