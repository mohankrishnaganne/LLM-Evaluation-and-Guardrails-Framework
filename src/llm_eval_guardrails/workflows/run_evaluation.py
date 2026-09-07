r"""Automated evaluation workflow: ingest, score, aggregate, publish.

This is the framework's batch entry point, installed as the ``llm-eval``
console script. It:

1. loads a sweep definition (or synthesises a single default configuration),
2. reads each configuration's dataset from S3 or a local path,
3. runs the evaluation suite and guardrails over every sample,
4. aggregates the results into a comparison summary, and
5. writes JSON, CSV and Markdown artefacts back to S3 or local disk.

The process exit code is a regression gate suitable for CI:

* ``0`` -- every configured threshold was met.
* ``1`` -- the evaluation ran but a quality gate failed (a metric fell below
  its threshold, or the failure rate exceeded ``max_failure_rate``).
* ``2`` -- the run could not complete (bad configuration, unreadable dataset,
  provider outage).

Distinguishing 1 from 2 matters: a failing gate should block a deployment,
whereas an infrastructure error should page someone instead of being mistaken
for a quality regression.

Example:
    llm-eval --dataset s3://my-bucket/eval/golden.jsonl \\
             --output s3://my-bucket/eval-reports \\
             --metrics faithfulness,answer_relevance
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..aws.dataset import DatasetLoader
from ..aws.s3 import S3Storage
from ..config import Settings, get_settings
from ..evaluators.suite import EvaluationSuite
from ..exceptions import ConfigurationError, DatasetError, FrameworkError, StorageError
from ..llm.registry import build_llm_client
from ..logging_config import bind_run_context, configure_logging, get_logger, new_correlation_id
from ..models import EvaluationReport, MetricName, RunConfig
from .reporting import SweepSummary, render_markdown, summarize_sweep

__all__ = ["ExitCode", "main", "run_sweep"]

_log = get_logger(__name__)


class ExitCode:
    """Process exit codes with CI-meaningful semantics."""

    SUCCESS = 0
    QUALITY_GATE_FAILED = 1
    EXECUTION_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Run RAG evaluation and guardrails across one or more configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        help="Dataset URI (s3://bucket/key or local path). Required unless --configs is given.",
    )
    parser.add_argument(
        "--configs",
        help=(
            "JSON file describing RAG configurations to sweep. Either a list of "
            "RunConfig objects or an object with a 'configs' key."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Destination prefix for report artefacts (s3://bucket/prefix or a local "
            "directory). Defaults to the configured report bucket."
        ),
    )
    parser.add_argument(
        "--metrics",
        help="Comma-separated metrics to run, overriding configuration.",
    )
    parser.add_argument(
        "--limit", type=int, help="Evaluate at most this many samples per configuration."
    )
    parser.add_argument("--concurrency", type=int, help="Override the maximum concurrent samples.")
    parser.add_argument("--run-id", help="Correlation id for this run; generated when omitted.")
    parser.add_argument(
        "--strict-dataset",
        action="store_true",
        help="Fail on the first malformed dataset record instead of skipping it.",
    )
    parser.add_argument(
        "--no-guardrails", action="store_true", help="Disable all guardrail checks."
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Embed per-sample detail in the published JSON summary.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        help=(
            "Fail the run if any metric's mean falls below this value, overriding "
            "the per-metric configured thresholds."
        ),
    )
    parser.add_argument("--log-level", help="Override the configured log level.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate datasets and configuration, then exit without scoring.",
    )
    return parser


def load_configs(uri: str, storage: S3Storage) -> list[RunConfig]:
    """Load RAG configurations for a sweep.

    Args:
        uri: JSON file URI or local path.
        storage: Storage wrapper used to read the file.

    Returns:
        The parsed configurations.

    Raises:
        ConfigurationError: If the file is not valid JSON or does not describe
            a list of configurations.
    """
    raw = storage.read_text(uri)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"configuration file {uri} is not valid JSON: {exc}"
        raise ConfigurationError(msg) from exc

    if isinstance(payload, dict):
        payload = payload.get("configs", payload.get("configurations"))
    if not isinstance(payload, list):
        msg = (
            f"configuration file {uri} must contain a JSON list of configurations "
            "(or an object with a 'configs' key)"
        )
        raise ConfigurationError(msg)

    try:
        return [RunConfig.model_validate(entry) for entry in payload]
    except ValidationError as exc:
        msg = f"invalid RAG configuration in {uri}: {exc}"
        raise ConfigurationError(msg) from exc


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply command-line overrides on top of environment configuration.

    Args:
        settings: The settings loaded from the environment.
        args: Parsed command-line arguments.

    Returns:
        A new settings object with overrides applied.

    Raises:
        ConfigurationError: If an override value is invalid.
    """
    updated = settings.model_copy(deep=True)

    if args.metrics:
        names = [n.strip() for n in args.metrics.split(",") if n.strip()]
        try:
            updated.evaluation.metrics = [MetricName(name) for name in names]
        except ValueError as exc:
            msg = "unknown metric in --metrics: {error} (valid: {valid})".format(
                error=exc, valid=", ".join(m.value for m in MetricName)
            )
            raise ConfigurationError(msg) from exc

    if args.concurrency:
        updated.evaluation.max_concurrency = args.concurrency
    if args.include_samples:
        updated.evaluation.include_samples_in_report = True
    if args.no_guardrails:
        updated.guardrails.enabled = False
    if args.fail_under is not None:
        updated.evaluation.thresholds = dict.fromkeys(updated.evaluation.metrics, args.fail_under)
    return updated


def _artifact_uri(output: str, run_id: str, filename: str) -> str:
    """Build a destination URI for one report artefact.

    Args:
        output: The output prefix (``s3://bucket/prefix`` or a local directory).
        run_id: The run's correlation id.
        filename: The artefact filename.

    Returns:
        The full destination URI or path.
    """
    prefix = output.rstrip("/")
    if prefix.startswith("s3://"):
        return f"{prefix}/{run_id}/{filename}"
    return str(Path(prefix) / run_id / filename)


def evaluate_gates(summary: SweepSummary, settings: Settings) -> list[str]:
    """Check the sweep against configured quality gates.

    Args:
        summary: The completed sweep.
        settings: Settings supplying thresholds and the failure-rate ceiling.

    Returns:
        Human-readable descriptions of every gate that failed; empty when the
        sweep passed.
    """
    failures: list[str] = []

    for report in summary.reports:
        for metric, aggregate in report.aggregates.items():
            threshold = settings.evaluation.threshold_for(metric)
            if threshold is None or aggregate.mean is None:
                continue
            if aggregate.mean < threshold:
                failures.append(
                    f"{report.config_name}: {metric.value} mean "
                    f"{aggregate.mean:.3f} < threshold {threshold:.3f}"
                )

        if report.failure_rate > settings.evaluation.max_failure_rate:
            failures.append(
                f"{report.config_name}: failure rate {report.failure_rate:.1%} "
                f"exceeds ceiling {settings.evaluation.max_failure_rate:.1%}"
            )

    return failures


async def run_sweep(
    settings: Settings,
    configs: Sequence[RunConfig],
    *,
    default_dataset: str | None,
    run_id: str,
    limit: int | None = None,
    strict_dataset: bool = False,
    dry_run: bool = False,
) -> SweepSummary:
    """Evaluate every configuration and aggregate the results.

    Configurations are evaluated sequentially rather than in parallel. Each one
    already saturates the judge's concurrency budget internally, so overlapping
    them would only contend for the same rate limit while making per-config
    latency numbers meaningless.

    Args:
        settings: The resolved application settings.
        configs: The RAG configurations to evaluate.
        default_dataset: Dataset URI used by configurations with no override.
        run_id: Correlation id for the sweep.
        limit: Evaluate at most this many samples per configuration.
        strict_dataset: Fail on the first malformed dataset record.
        dry_run: Load and validate inputs, then return without scoring.

    Returns:
        The aggregated sweep summary. On a dry run it contains no reports.

    Raises:
        ConfigurationError: If a configuration has no dataset to evaluate.
        DatasetError: If a dataset cannot be read or produces no valid samples.
    """
    loader = DatasetLoader(settings.aws)
    llm = build_llm_client(settings)
    reports: list[EvaluationReport] = []

    try:
        suite = EvaluationSuite(llm, settings)

        for config in configs:
            dataset_uri = config.dataset_uri or default_dataset
            if not dataset_uri:
                msg = f"configuration {config.name!r} has no dataset_uri and no --dataset was given"
                raise ConfigurationError(msg)

            samples = await loader.aload(dataset_uri, strict=strict_dataset, limit=limit)
            _log.info(
                "sweep.config_loaded",
                config_name=config.name,
                dataset_uri=dataset_uri,
                sample_count=len(samples),
            )

            if dry_run:
                continue

            resolved = config.model_copy(update={"dataset_uri": dataset_uri})
            report = await suite.aevaluate_dataset(samples, resolved, run_id=run_id)
            reports.append(report)
    finally:
        await llm.aclose()

    return summarize_sweep(run_id, reports)


def publish(
    summary: SweepSummary,
    settings: Settings,
    *,
    output: str,
    include_samples: bool,
) -> dict[str, str]:
    """Write the sweep artefacts to S3 or local disk.

    Failures to publish one artefact do not prevent the others from being
    written: a CI job that produced results should surface as many of them as
    possible even if, say, the CSV write races a bucket policy change.

    Args:
        summary: The completed sweep.
        settings: Settings supplying AWS configuration.
        output: Destination prefix.
        include_samples: Embed per-sample detail in the JSON summary.

    Returns:
        A mapping of artefact name to the URI written, omitting failures.
    """
    storage = S3Storage(settings.aws)
    written: dict[str, str] = {}

    artefacts: list[tuple[str, str, Any]] = [
        ("summary_json", "summary.json", summary.to_dict(include_samples=include_samples)),
        ("summary_csv", "summary.csv", summary.summary_rows()),
        ("report_markdown", "report.md", render_markdown(summary)),
    ]

    for name, filename, payload in artefacts:
        uri = _artifact_uri(output, summary.run_id, filename)
        try:
            if filename.endswith(".json"):
                written[name] = storage.write_json(uri, payload)
            elif filename.endswith(".csv"):
                written[name] = storage.write_csv(uri, payload)
            else:
                written[name] = storage.write_text(uri, payload, content_type="text/markdown")
        except (StorageError, DatasetError) as exc:
            _log.error("report.publish_failed", artifact=name, uri=uri, error=str(exc))

    for report in summary.reports:
        uri = _artifact_uri(output, summary.run_id, f"detail-{_slug(report.config_name)}.json")
        try:
            storage.write_json(uri, report.model_dump(mode="json"))
            written[f"detail_{_slug(report.config_name)}"] = uri
        except StorageError as exc:
            _log.error("report.publish_failed", artifact=report.config_name, error=str(exc))

    return written


def _slug(value: str) -> str:
    """Convert a configuration name into a filename-safe slug.

    Args:
        value: The raw name.

    Returns:
        The slug, with unsafe characters replaced by hyphens.
    """
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-") or "config"


async def _amain(argv: Sequence[str] | None = None) -> int:
    """Run the workflow end to end.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        The process exit code.
    """
    args = build_parser().parse_args(argv)

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        # Logging is not configured yet, so report on stderr directly.
        print(f"configuration error: {exc}", file=sys.stderr)
        return ExitCode.EXECUTION_ERROR

    configure_logging(
        level=args.log_level or settings.log_level,
        json_output=settings.json_logs,
        force=True,
    )
    run_id = args.run_id or new_correlation_id()
    bind_run_context(run_id=run_id)

    try:
        settings = apply_overrides(settings, args)

        storage = S3Storage(settings.aws)
        configs = (
            load_configs(args.configs, storage)
            if args.configs
            else [RunConfig(name="default", dataset_uri=args.dataset)]
        )
        if not configs:
            msg = "no RAG configurations to evaluate"
            raise ConfigurationError(msg)
        if not args.dataset and not any(c.dataset_uri for c in configs):
            msg = "either --dataset or a dataset_uri on every configuration is required"
            raise ConfigurationError(msg)

        _log.info(
            "workflow.started",
            run_id=run_id,
            config_count=len(configs),
            metrics=[m.value for m in settings.evaluation.metrics],
            guardrails_enabled=settings.guardrails.enabled,
            provider=str(settings.llm.provider),
            dry_run=args.dry_run,
        )

        summary = await run_sweep(
            settings,
            configs,
            default_dataset=args.dataset,
            run_id=run_id,
            limit=args.limit,
            strict_dataset=args.strict_dataset,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            _log.info("workflow.dry_run_complete", run_id=run_id, config_count=len(configs))
            return ExitCode.SUCCESS

        output = args.output or (
            "s3://{bucket}/{prefix}".format(
                bucket=settings.aws.report_bucket,
                prefix=settings.aws.report_prefix.strip("/"),
            )
            if settings.aws.report_bucket
            else "./reports"
        )
        written = publish(summary, settings, output=output, include_samples=args.include_samples)

        gate_failures = evaluate_gates(summary, settings)
        _log.info(
            "workflow.completed",
            run_id=run_id,
            config_count=len(summary.reports),
            total_samples=sum(r.sample_count for r in summary.reports),
            artifacts=written,
            gate_failures=gate_failures,
        )

        print(render_markdown(summary))

        if gate_failures:
            for failure in gate_failures:
                print(f"QUALITY GATE FAILED: {failure}", file=sys.stderr)
            return ExitCode.QUALITY_GATE_FAILED
        return ExitCode.SUCCESS

    except FrameworkError as exc:
        _log.error(
            "workflow.failed",
            run_id=run_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return ExitCode.EXECUTION_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive interruption
        _log.warning("workflow.interrupted", run_id=run_id)
        return ExitCode.EXECUTION_ERROR
    except Exception:  # noqa: BLE001 - top-level boundary must not leak a traceback
        _log.exception("workflow.unexpected_error", run_id=run_id)
        print("evaluation failed with an unexpected error", file=sys.stderr)
        return ExitCode.EXECUTION_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        The process exit code.
    """
    return asyncio.run(_amain(argv))


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    sys.exit(main())
