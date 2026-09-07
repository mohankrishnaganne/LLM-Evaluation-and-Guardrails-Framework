"""Structured JSON logging configured for AWS CloudWatch ingestion.

CloudWatch Logs Insights parses single-line JSON events natively, so every log
record is emitted as one compact JSON object on stdout. Container platforms
(ECS/Fargate, EKS, Lambda) forward stdout to CloudWatch without a sidecar.

Typical usage::

    from llm_eval_guardrails.logging_config import configure_logging, get_logger

    configure_logging()
    log = get_logger(__name__)
    log.info("evaluation.completed", metric="faithfulness", score=0.91)
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

import structlog
from structlog.typing import EventDict, Processor, WrappedLogger

__all__ = [
    "bind_run_context",
    "configure_logging",
    "get_logger",
    "new_correlation_id",
    "request_context",
]

_CONFIGURED: bool = False

#: Keys promoted to the top level of every event so CloudWatch queries such as
#: ``fields @timestamp, level, event`` work without nested accessors.
_SERVICE_KEY: Final[str] = "service"


def _add_service_metadata(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Attach static deployment metadata to every event.

    Args:
        _logger: Unused wrapped logger (structlog processor signature).
        _method: Unused log method name.
        event_dict: The mutable event payload.

    Returns:
        The event dict enriched with service, environment and version fields.
    """
    event_dict.setdefault(_SERVICE_KEY, os.getenv("SERVICE_NAME", "llm-eval-guardrails"))
    event_dict.setdefault("env", os.getenv("DEPLOY_ENV", "local"))
    version = os.getenv("SERVICE_VERSION")
    if version:
        event_dict.setdefault("version", version)
    return event_dict


def _rename_event_to_message(
    _logger: WrappedLogger, _method: str, event_dict: EventDict
) -> EventDict:
    """Duplicate ``event`` into ``message`` for CloudWatch/OpenSearch conventions.

    ``event`` is retained because existing dashboards and the framework's own
    tests key off it; ``message`` is added because most log-aggregation UIs
    render that field by default.

    Args:
        _logger: Unused wrapped logger.
        _method: Unused log method name.
        event_dict: The mutable event payload.

    Returns:
        The event dict with a ``message`` mirror of ``event``.
    """
    event = event_dict.get("event")
    if isinstance(event, str):
        event_dict.setdefault("message", event)
    return event_dict


def configure_logging(
    *,
    level: str | int = "INFO",
    json_output: bool | None = None,
    force: bool = False,
) -> None:
    """Configure structlog and the stdlib logging bridge exactly once.

    Calling this more than once is a no-op unless ``force`` is set, which keeps
    library imports and test fixtures from clobbering an application's chosen
    configuration.

    Args:
        level: Minimum level to emit, as a name (``"INFO"``) or numeric value.
        json_output: Emit JSON when ``True`` and a coloured console renderer when
            ``False``. When ``None`` (the default), JSON is used unless stdout is
            an interactive TTY, which keeps local development readable while
            guaranteeing machine-parseable output in containers.
        force: Reconfigure even if logging was already configured.
    """
    global _CONFIGURED  # noqa: PLW0603 - module-level idempotency guard
    if _CONFIGURED and not force:
        return

    if json_output is None:
        json_output = not sys.stdout.isatty()

    numeric_level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):  # pragma: no cover - defensive
        numeric_level = logging.INFO

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_service_metadata,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Processor
    if json_output:
        shared.append(_rename_event_to_message)
        shared.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(numeric_level)

    # boto3/botocore are extremely chatty at DEBUG and leak signed headers.
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(numeric_level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring logging on first use.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A bound logger that emits structured events.
    """
    if not _CONFIGURED:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def new_correlation_id() -> str:
    """Generate a fresh correlation identifier.

    Returns:
        A hex UUID4 string suitable for tracing one request or evaluation run.
    """
    return uuid.uuid4().hex


def bind_run_context(**kwargs: Any) -> None:
    """Bind key/value pairs to the ambient logging context for this task.

    The values propagate to every subsequent log event emitted on the same
    thread or asyncio task, including inside library code.

    Args:
        **kwargs: Fields to bind, e.g. ``run_id="abc"``.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


@contextmanager
def request_context(**kwargs: Any) -> Iterator[str]:
    """Scope a correlation id (and extra fields) to a block of work.

    Any ``correlation_id`` supplied in ``kwargs`` is used verbatim; otherwise a
    new one is generated. The context is fully unwound on exit, including when
    the block raises.

    Args:
        **kwargs: Additional fields to bind for the duration of the block.

    Yields:
        The correlation id bound for the block.
    """
    correlation_id = str(kwargs.pop("correlation_id", None) or new_correlation_id())
    tokens = structlog.contextvars.bind_contextvars(correlation_id=correlation_id, **kwargs)
    try:
        yield correlation_id
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
