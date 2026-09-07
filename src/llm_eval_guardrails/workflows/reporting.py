"""Aggregation and rendering of multi-configuration evaluation results.

A sweep produces one :class:`~llm_eval_guardrails.models.EvaluationReport` per
RAG configuration. This module combines them into the artefacts a team actually
consumes:

* a machine-readable JSON bundle for dashboards and regression gates,
* a flat CSV of per-configuration summary rows for spreadsheets, and
* a Markdown comparison table for pull-request comments and Slack.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from ..models import EvaluationReport, MetricName

__all__ = ["SweepSummary", "render_markdown", "summarize_sweep"]


class SweepSummary:
    """The combined outcome of evaluating several RAG configurations.

    Attributes:
        run_id: Correlation id shared by every configuration in the sweep.
        reports: The per-configuration reports, in execution order.
        generated_at: UTC timestamp at which the summary was assembled.
    """

    def __init__(self, run_id: str, reports: Sequence[EvaluationReport]) -> None:
        """Initialise the summary.

        Args:
            run_id: Correlation id for the sweep.
            reports: The per-configuration reports.
        """
        self.run_id = run_id
        self.reports = list(reports)
        self.generated_at = datetime.now(timezone.utc)

    @property
    def metrics(self) -> list[MetricName]:
        """Every metric present in any report, in first-seen order.

        Returns:
            The metric names.
        """
        seen: dict[MetricName, None] = {}
        for report in self.reports:
            for metric in report.aggregates:
                seen.setdefault(metric, None)
        return list(seen)

    def summary_rows(self) -> list[dict[str, Any]]:
        """Flatten every report into one row per configuration.

        Returns:
            The summary rows, suitable for CSV output.
        """
        return [report.summary_row() for report in self.reports]

    def best_config(self, metric: MetricName) -> str | None:
        """Identify the configuration with the highest mean for one metric.

        Args:
            metric: The metric to rank by.

        Returns:
            The winning configuration's name, or ``None`` when no report
            produced a mean for that metric.
        """
        best_name: str | None = None
        best_value = float("-inf")
        for report in self.reports:
            aggregate = report.aggregates.get(metric)
            if aggregate is None or aggregate.mean is None:
                continue
            if aggregate.mean > best_value:
                best_value = aggregate.mean
                best_name = report.config_name
        return best_name

    def total_failure_rate(self) -> float:
        """Compute the sample-weighted failure rate across the whole sweep.

        Returns:
            The weighted failure rate in [0, 1]; ``0.0`` when nothing ran.
        """
        total_samples = sum(r.sample_count for r in self.reports)
        if total_samples == 0:
            return 0.0
        weighted = sum(r.failure_rate * r.sample_count for r in self.reports)
        return weighted / total_samples

    def to_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        """Serialise the sweep for JSON output.

        Args:
            include_samples: Embed per-sample detail in each report. Omitting
                it keeps the summary artefact small enough to read at a glance.

        Returns:
            The JSON-serialisable payload.
        """
        reports = [
            (report if include_samples else report.without_samples()).model_dump(mode="json")
            for report in self.reports
        ]
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at.isoformat(),
            "config_count": len(self.reports),
            "total_samples": sum(r.sample_count for r in self.reports),
            "total_llm_calls": sum(r.llm_calls for r in self.reports),
            "total_tokens": sum(r.total_tokens for r in self.reports),
            "total_blocked_samples": sum(r.blocked_samples for r in self.reports),
            "weighted_failure_rate": round(self.total_failure_rate(), 4),
            "best_by_metric": {metric.value: self.best_config(metric) for metric in self.metrics},
            "guardrail_triggers": self.merged_triggers(),
            "reports": reports,
        }

    def merged_triggers(self) -> dict[str, int]:
        """Sum guardrail trigger counts across every configuration.

        Returns:
            A mapping of ``"<stage>:<rule_id>"`` to total occurrences.
        """
        merged: dict[str, int] = {}
        for report in self.reports:
            for key, count in report.guardrail_triggers.items():
                merged[key] = merged.get(key, 0) + count
        return merged


def summarize_sweep(run_id: str, reports: Sequence[EvaluationReport]) -> SweepSummary:
    """Build a :class:`SweepSummary` from per-configuration reports.

    Args:
        run_id: Correlation id for the sweep.
        reports: The per-configuration reports.

    Returns:
        The assembled summary.
    """
    return SweepSummary(run_id, reports)


def render_markdown(summary: SweepSummary) -> str:
    """Render the sweep as a Markdown comparison table.

    Args:
        summary: The sweep to render.

    Returns:
        A Markdown document with a per-configuration metric table, guardrail
        trigger counts and the winning configuration per metric.
    """
    metrics = summary.metrics
    lines: list[str] = [
        "# RAG Evaluation Report",
        "",
        f"- **Run ID:** `{summary.run_id}`",
        "- **Generated:** {ts}".format(ts=summary.generated_at.isoformat(timespec="seconds")),
        f"- **Configurations:** {len(summary.reports)}",
        f"- **Total samples:** {sum(r.sample_count for r in summary.reports)}",
        f"- **Weighted failure rate:** {summary.total_failure_rate():.2%}",
        "",
        "## Metric comparison",
        "",
    ]

    if not summary.reports:
        lines.append("_No configurations were evaluated._")
        return "\n".join(lines)

    header = ["Configuration", "Samples"]
    for metric in metrics:
        header.append(metric.value.replace("_", " ").title())
    header.extend(["Blocked", "Failures"])

    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for report in summary.reports:
        row = [report.config_name, str(report.sample_count)]
        for metric in metrics:
            aggregate = report.aggregates.get(metric)
            if aggregate is None or aggregate.mean is None:
                row.append("-")
                continue
            cell = f"{aggregate.mean:.3f}"
            if aggregate.pass_rate is not None:
                cell += f" ({aggregate.pass_rate:.0%} pass)"
            row.append(cell)
        row.append(str(report.blocked_samples))
        row.append(f"{report.failure_rate:.1%}")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Best configuration per metric", ""])
    for metric in metrics:
        winner = summary.best_config(metric)
        lines.append(
            "- **{metric}**: {winner}".format(
                metric=metric.value.replace("_", " ").title(),
                winner=winner or "_no result_",
            )
        )

    triggers = summary.merged_triggers()
    lines.extend(["", "## Guardrail triggers", ""])
    if not triggers:
        lines.append("_No guardrails were triggered._")
    else:
        lines.append("| Rule | Count |")
        lines.append("|---|---|")
        for rule, count in sorted(triggers.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| `{rule}` | {count} |")

    return "\n".join(lines) + "\n"
