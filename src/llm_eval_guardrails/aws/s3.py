"""S3 integration for reading evaluation datasets and writing reports.

The module exposes a thin, testable wrapper over boto3 rather than passing raw
clients around. Everything is addressed by URI (``s3://bucket/key`` or a local
path), so the same workflow code runs against S3 in production and the local
filesystem in tests and development.

All blocking boto3 calls are dispatched to a thread so the async workflow never
stalls its event loop.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..config import AWSSettings
from ..exceptions import DatasetError, StorageError
from ..logging_config import get_logger

#: botocore synthesises operation methods (``get_object``, ``put_object``, ...)
#: at runtime from service JSON, so no static type describes them. ``Any`` is
#: the honest annotation; the alias keeps that decision explicit and greppable
#: rather than scattering bare ``Any`` through the module.
S3Client = Any

__all__ = ["S3Location", "S3Storage", "is_s3_uri", "parse_s3_uri"]

_log = get_logger(__name__)

_S3_SCHEME: Final[str] = "s3://"

#: Refuse to buffer objects larger than this into memory. Evaluation datasets
#: are text and normally measured in megabytes; a multi-gigabyte object almost
#: always means the wrong key was configured.
_MAX_OBJECT_BYTES: Final[int] = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class S3Location:
    """A parsed ``s3://bucket/key`` location.

    Attributes:
        bucket: The bucket name.
        key: The object key.
    """

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        """Render the location back to an ``s3://`` URI.

        Returns:
            The canonical URI.
        """
        return f"{_S3_SCHEME}{self.bucket}/{self.key}"


def is_s3_uri(uri: str) -> bool:
    """Report whether a URI addresses S3.

    Args:
        uri: The candidate URI or path.

    Returns:
        ``True`` when the URI uses the ``s3://`` scheme.
    """
    return uri.startswith(_S3_SCHEME)


def parse_s3_uri(uri: str) -> S3Location:
    """Parse an ``s3://bucket/key`` URI.

    Args:
        uri: The URI to parse.

    Returns:
        The parsed location.

    Raises:
        StorageError: If the URI is not a well-formed S3 URI with both a
            bucket and a non-empty key.
    """
    if not is_s3_uri(uri):
        msg = f"not an s3:// URI: {uri!r}"
        raise StorageError(msg)
    remainder = uri[len(_S3_SCHEME) :]
    bucket, separator, key = remainder.partition("/")
    if not bucket or not separator or not key:
        msg = f"malformed S3 URI (expected s3://bucket/key): {uri!r}"
        raise StorageError(msg)
    return S3Location(bucket=bucket, key=key)


class S3Storage:
    """Reads and writes evaluation artefacts on S3 or the local filesystem.

    Local paths are handled natively rather than requiring a mock S3, which
    keeps development loops fast and lets CI exercise the same code paths
    without network access.

    Example:
        >>> storage = S3Storage(aws_settings)  # doctest: +SKIP
        >>> rows = await storage.aread_jsonl("s3://my-bucket/eval/dataset.jsonl")  # doctest: +SKIP
    """

    def __init__(self, settings: AWSSettings, *, client: S3Client | None = None) -> None:
        """Initialise the storage wrapper.

        Args:
            settings: AWS region, profile, endpoint and encryption settings.
            client: Pre-built S3 client, primarily for tests. When omitted, the
                client is created lazily on first S3 access so that purely
                local runs never require AWS credentials.
        """
        self._settings = settings
        self._client = client

    @property
    def client(self) -> S3Client:
        """The underlying S3 client, created on first use.

        Returns:
            A configured boto3 S3 client.

        Raises:
            StorageError: If the client could not be created.
        """
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> S3Client:
        """Construct a configured S3 client.

        Returns:
            The boto3 client.

        Raises:
            StorageError: If the AWS SDK could not create the client.
        """
        import boto3
        from botocore.config import Config

        try:
            session = boto3.Session(
                profile_name=self._settings.profile, region_name=self._settings.region
            )
            client: S3Client = session.client(
                "s3",
                endpoint_url=self._settings.endpoint_url,
                config=Config(
                    region_name=self._settings.region,
                    retries={"max_attempts": 5, "mode": "adaptive"},
                    max_pool_connections=self._settings.max_pool_connections,
                    user_agent_extra="llm-eval-guardrails/0.1.0",
                ),
            )
        except Exception as exc:
            msg = f"failed to create S3 client: {exc}"
            raise StorageError(msg) from exc
        return client

    # ------------------------------------------------------------------ #
    # Raw text I/O
    # ------------------------------------------------------------------ #

    def read_text(self, uri: str) -> str:
        """Read an object or local file as UTF-8 text.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.

        Returns:
            The decoded contents.

        Raises:
            StorageError: If the object is missing, too large, or unreadable.
        """
        if not is_s3_uri(uri):
            return self._read_local(uri)

        location = parse_s3_uri(uri)
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            response = self.client.get_object(Bucket=location.bucket, Key=location.key)
            size = int(response.get("ContentLength", 0) or 0)
            if size > _MAX_OBJECT_BYTES:
                msg = f"object {uri} is {size} bytes, exceeding the {_MAX_OBJECT_BYTES}-byte limit"
                raise StorageError(msg)
            body: bytes = response["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            msg = f"failed to read {uri} [{code}]: {exc}"
            raise StorageError(msg) from exc
        except BotoCoreError as exc:
            msg = f"transport failure reading {uri}: {exc}"
            raise StorageError(msg) from exc

        _log.debug("s3.read", uri=uri, bytes=len(body))
        return body.decode("utf-8")

    @staticmethod
    def _read_local(path: str) -> str:
        """Read a local file as UTF-8 text.

        Args:
            path: The filesystem path.

        Returns:
            The decoded contents.

        Raises:
            StorageError: If the file is missing or unreadable.
        """
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"failed to read local file {path}: {exc}"
            raise StorageError(msg) from exc

    async def aread_text(self, uri: str) -> str:
        """Async wrapper over :meth:`read_text`.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.

        Returns:
            The decoded contents.
        """
        return await asyncio.to_thread(self.read_text, uri)

    def write_text(self, uri: str, content: str, *, content_type: str = "application/json") -> str:
        """Write UTF-8 text to S3 or the local filesystem.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.
            content: The text to write.
            content_type: MIME type recorded on the S3 object.

        Returns:
            The URI or path written.

        Raises:
            StorageError: If the write failed.
        """
        payload = content.encode("utf-8")

        if not is_s3_uri(uri):
            try:
                destination = Path(uri)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            except OSError as exc:
                msg = f"failed to write local file {uri}: {exc}"
                raise StorageError(msg) from exc
            _log.info("storage.wrote_local", path=uri, bytes=len(payload))
            return uri

        location = parse_s3_uri(uri)
        from botocore.exceptions import BotoCoreError, ClientError

        request: dict[str, Any] = {
            "Bucket": location.bucket,
            "Key": location.key,
            "Body": payload,
            "ContentType": content_type,
        }
        if self._settings.server_side_encryption:
            request["ServerSideEncryption"] = self._settings.server_side_encryption
            if self._settings.kms_key_id:
                request["SSEKMSKeyId"] = self._settings.kms_key_id

        try:
            self.client.put_object(**request)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            msg = f"failed to write {uri} [{code}]: {exc}"
            raise StorageError(msg) from exc
        except BotoCoreError as exc:
            msg = f"transport failure writing {uri}: {exc}"
            raise StorageError(msg) from exc

        _log.info("s3.wrote", uri=uri, bytes=len(payload))
        return uri

    async def awrite_text(
        self, uri: str, content: str, *, content_type: str = "application/json"
    ) -> str:
        """Async wrapper over :meth:`write_text`.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.
            content: The text to write.
            content_type: MIME type recorded on the S3 object.

        Returns:
            The URI or path written.
        """
        return await asyncio.to_thread(self.write_text, uri, content, content_type=content_type)

    # ------------------------------------------------------------------ #
    # Structured formats
    # ------------------------------------------------------------------ #

    def read_jsonl(self, uri: str) -> list[dict[str, Any]]:
        """Read newline-delimited JSON records.

        Blank lines are skipped. A malformed line reports its own line number,
        because in a ten-thousand-row dataset "invalid JSON" alone is not a
        diagnosis.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.

        Returns:
            The decoded records.

        Raises:
            DatasetError: If any line is not a JSON object.
        """
        return list(_iter_jsonl(self.read_text(uri), uri))

    async def aread_jsonl(self, uri: str) -> list[dict[str, Any]]:
        """Async wrapper over :meth:`read_jsonl`.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.

        Returns:
            The decoded records.
        """
        return await asyncio.to_thread(self.read_jsonl, uri)

    def read_csv(self, uri: str) -> list[dict[str, Any]]:
        """Read a CSV file with a header row into dictionaries.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.

        Returns:
            One dictionary per data row, keyed by header name.

        Raises:
            DatasetError: If the file has no header row.
        """
        text = self.read_text(uri)
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            msg = f"CSV dataset {uri} has no header row"
            raise DatasetError(msg)
        return [dict(row) for row in reader]

    async def aread_csv(self, uri: str) -> list[dict[str, Any]]:
        """Async wrapper over :meth:`read_csv`.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.

        Returns:
            One dictionary per data row.
        """
        return await asyncio.to_thread(self.read_csv, uri)

    def write_json(self, uri: str, payload: Any, *, indent: int | None = 2) -> str:
        """Serialise a payload as JSON and write it.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.
            payload: Any JSON-serialisable object.
            indent: Indentation for readability; ``None`` writes compact JSON.

        Returns:
            The URI or path written.
        """
        content = json.dumps(payload, indent=indent, default=str, ensure_ascii=False)
        return self.write_text(uri, content, content_type="application/json")

    async def awrite_json(self, uri: str, payload: Any, *, indent: int | None = 2) -> str:
        """Async wrapper over :meth:`write_json`.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.
            payload: Any JSON-serialisable object.
            indent: Indentation for readability.

        Returns:
            The URI or path written.
        """
        return await asyncio.to_thread(self.write_json, uri, payload, indent=indent)

    def write_csv(
        self, uri: str, rows: Sequence[dict[str, Any]], *, fieldnames: Sequence[str] | None = None
    ) -> str:
        """Write dictionaries as CSV.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.
            rows: The rows to write.
            fieldnames: Explicit column order. When omitted, the union of all
                row keys is used, ordered by first appearance so that the
                columns of the first row stay leftmost.

        Returns:
            The URI or path written.

        Raises:
            DatasetError: If ``rows`` is empty and no ``fieldnames`` were given.
        """
        if fieldnames is None:
            ordered: dict[str, None] = {}
            for row in rows:
                for key in row:
                    ordered.setdefault(key, None)
            fieldnames = list(ordered)
        if not fieldnames:
            msg = f"cannot write CSV to {uri}: no columns and no rows"
            raise DatasetError(msg)

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return self.write_text(uri, buffer.getvalue(), content_type="text/csv")

    async def awrite_csv(
        self, uri: str, rows: Sequence[dict[str, Any]], *, fieldnames: Sequence[str] | None = None
    ) -> str:
        """Async wrapper over :meth:`write_csv`.

        Args:
            uri: An ``s3://bucket/key`` URI or a local path.
            rows: The rows to write.
            fieldnames: Explicit column order.

        Returns:
            The URI or path written.
        """
        return await asyncio.to_thread(self.write_csv, uri, rows, fieldnames=fieldnames)

    def report_uri(self, run_id: str, config_name: str, suffix: str) -> str:
        """Build a conventional S3 destination for a report artefact.

        Args:
            run_id: The run's correlation id.
            config_name: The configuration the report describes.
            suffix: File suffix, e.g. ``"report.json"``.

        Returns:
            The destination URI.

        Raises:
            StorageError: If no report bucket is configured.
        """
        if not self._settings.report_bucket:
            msg = "no report bucket configured (set LEG_AWS__REPORT_BUCKET)"
            raise StorageError(msg)
        safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in config_name)
        return "{scheme}{bucket}/{prefix}/{run}/{name}.{suffix}".format(
            scheme=_S3_SCHEME,
            bucket=self._settings.report_bucket,
            prefix=self._settings.report_prefix.strip("/"),
            run=run_id,
            name=safe_name,
            suffix=suffix,
        )


def _iter_jsonl(text: str, uri: str) -> Iterator[dict[str, Any]]:
    """Parse newline-delimited JSON, reporting the offending line on failure.

    Args:
        text: The raw file contents.
        uri: Source URI, used in error messages.

    Yields:
        One decoded object per non-blank line.

    Raises:
        DatasetError: If a line is invalid JSON or is not a JSON object.
    """
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"invalid JSON at {uri}:{line_number}: {exc}"
            raise DatasetError(msg) from exc
        if not isinstance(record, dict):
            msg = f"expected a JSON object at {uri}:{line_number}, got {type(record).__name__}"
            raise DatasetError(msg)
        yield record
