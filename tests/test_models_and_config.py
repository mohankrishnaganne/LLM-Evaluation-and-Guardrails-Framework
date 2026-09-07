"""Tests for domain models, aggregation and configuration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from llm_eval_guardrails.config import (
    EvaluationSettings,
    GuardrailSettings,
    LLMSettings,
    Settings,
    get_settings,
    reset_settings_cache,
)
from llm_eval_guardrails.exceptions import ConfigurationError
from llm_eval_guardrails.models import (
    AggregateMetric,
    EvaluationReport,
    MetricName,
    RAGSample,
    RetrievedDocument,
    SingleMetricResult,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TestRAGSample:
    def test_requires_a_question(self):
        with pytest.raises(ValidationError):
            RAGSample(sample_id="s", question="", answer="a")

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            # The invalid keyword is the point of the test: it proves that
            # extra="forbid" turns a dataset typo into a loud failure.
            RAGSample(sample_id="s", question="q?", answer="a", typo_field=1)  # type: ignore[call-arg]

    def test_context_texts(self, sample):
        assert sample.context_texts[0].startswith("Refunds are accepted")

    def test_joined_context_is_one_indexed(self, sample):
        assert sample.joined_context().startswith("[1] ")
        assert "[2] " in sample.joined_context()

    def test_joined_context_sentinel_when_empty(self):
        bare = RAGSample(sample_id="s", question="q?", answer="a")
        assert bare.joined_context() == "[NO CONTEXT RETRIEVED]"

    def test_joined_context_truncates(self, sample):
        rendered = sample.joined_context(max_chars_per_doc=10)
        assert "[truncated]" in rendered

    def test_truncation_is_a_no_op_below_the_budget(self):
        doc = RetrievedDocument(doc_id="d", content="short")
        assert doc.truncated(100) == "short"

    def test_non_positive_budget_disables_truncation(self):
        doc = RetrievedDocument(doc_id="d", content="content here")
        assert doc.truncated(0) == "content here"

    def test_document_requires_non_empty_content(self):
        with pytest.raises(ValidationError):
            RetrievedDocument(doc_id="d", content="")


class TestSingleMetricResult:
    def test_succeeded_requires_a_score_and_no_error(self):
        ok = SingleMetricResult(metric=MetricName.FAITHFULNESS, sample_id="s", score=0.9)
        assert ok.succeeded

    def test_error_means_not_succeeded(self):
        failed = SingleMetricResult(
            metric=MetricName.FAITHFULNESS, sample_id="s", score=0.9, error="boom"
        )
        assert not failed.succeeded

    def test_missing_score_means_not_succeeded(self):
        skipped = SingleMetricResult(metric=MetricName.FAITHFULNESS, sample_id="s")
        assert not skipped.succeeded

    def test_score_must_be_within_zero_and_one(self):
        with pytest.raises(ValidationError):
            SingleMetricResult(metric=MetricName.FAITHFULNESS, sample_id="s", score=1.5)


class TestAggregateMetric:
    def _results(self, *scores: float) -> list[SingleMetricResult]:
        return [
            SingleMetricResult(metric=MetricName.FAITHFULNESS, sample_id=f"s{i}", score=s)
            for i, s in enumerate(scores)
        ]

    def test_computes_descriptive_statistics(self):
        agg = AggregateMetric.from_results(MetricName.FAITHFULNESS, self._results(0.0, 0.5, 1.0))
        assert agg.count == 3
        assert agg.mean == pytest.approx(0.5)
        assert agg.median == pytest.approx(0.5)
        assert agg.minimum == 0.0
        assert agg.maximum == 1.0

    def test_stdev_of_a_single_score_is_zero(self):
        agg = AggregateMetric.from_results(MetricName.FAITHFULNESS, self._results(0.5))
        assert agg.stdev == 0.0

    def test_percentiles_interpolate(self):
        agg = AggregateMetric.from_results(
            MetricName.FAITHFULNESS, self._results(0.0, 0.25, 0.5, 0.75, 1.0)
        )
        assert agg.p10 == pytest.approx(0.1)
        assert agg.p90 == pytest.approx(0.9)

    def test_pass_rate_uses_the_threshold(self):
        agg = AggregateMetric.from_results(
            MetricName.FAITHFULNESS, self._results(0.9, 0.9, 0.1), threshold=0.8
        )
        assert agg.pass_rate == pytest.approx(2 / 3)

    def test_score_equal_to_threshold_passes(self):
        agg = AggregateMetric.from_results(
            MetricName.FAITHFULNESS, self._results(0.8), threshold=0.8
        )
        assert agg.pass_rate == 1.0

    def test_pass_rate_is_none_without_a_threshold(self):
        assert (
            AggregateMetric.from_results(MetricName.FAITHFULNESS, self._results(0.5)).pass_rate
            is None
        )

    def test_failures_are_excluded_from_statistics_but_counted(self):
        results = [
            *self._results(1.0),
            SingleMetricResult(metric=MetricName.FAITHFULNESS, sample_id="x", error="boom"),
        ]
        agg = AggregateMetric.from_results(MetricName.FAITHFULNESS, results)
        assert agg.count == 1
        assert agg.failed == 1
        assert agg.mean == 1.0

    def test_skips_are_counted_separately_from_failures(self):
        results = [
            SingleMetricResult(
                metric=MetricName.CONTEXT_RECALL, sample_id="x", skipped_reason="no gt"
            )
        ]
        agg = AggregateMetric.from_results(MetricName.CONTEXT_RECALL, results)
        assert agg.skipped == 1
        assert agg.failed == 0
        assert agg.count == 0
        assert agg.mean is None

    def test_no_results_produces_an_empty_aggregate(self):
        agg = AggregateMetric.from_results(MetricName.FAITHFULNESS, [])
        assert agg.count == 0
        assert agg.mean is None


class TestEvaluationReport:
    def _report(self, **kwargs) -> EvaluationReport:
        start = utcnow()
        defaults = {
            "run_id": "r1",
            "config_name": "cfg",
            "started_at": start,
            "completed_at": start + timedelta(seconds=5),
            "sample_count": 2,
        }
        defaults.update(kwargs)
        return EvaluationReport(**defaults)

    def test_duration_is_computed(self):
        assert self._report().duration_seconds == pytest.approx(5.0)

    def test_rejects_completion_before_start(self):
        start = utcnow()
        with pytest.raises(ValidationError, match="must not precede"):
            EvaluationReport(
                run_id="r",
                config_name="c",
                started_at=start,
                completed_at=start - timedelta(seconds=1),
                sample_count=0,
            )

    def test_summary_row_flattens_metrics(self):
        report = self._report(
            aggregates={
                MetricName.FAITHFULNESS: AggregateMetric(
                    metric=MetricName.FAITHFULNESS,
                    count=2,
                    mean=0.875,
                    pass_rate=0.5,
                    threshold=0.8,
                )
            }
        )
        row = report.summary_row()
        assert row["faithfulness_mean"] == 0.875
        assert row["faithfulness_pass_rate"] == 0.5
        assert row["config_name"] == "cfg"

    def test_without_samples_strips_detail(self, sample):
        from llm_eval_guardrails.models import SampleResult

        report = self._report(sample_results=[SampleResult(sample_id="s-1")])
        assert report.sample_results
        assert report.without_samples().sample_results == []
        # The original is unchanged.
        assert report.sample_results

    def test_report_round_trips_through_json(self):
        report = self._report()
        restored = EvaluationReport.model_validate_json(report.model_dump_json())
        assert restored.run_id == report.run_id


class TestSettings:
    def test_defaults_are_valid(self):
        settings = Settings()
        assert settings.evaluation.metrics
        assert settings.guardrails.enabled

    def test_metrics_are_deduplicated_preserving_order(self):
        settings = EvaluationSettings(
            metrics=[
                MetricName.FAITHFULNESS,
                MetricName.CONTEXT_RECALL,
                MetricName.FAITHFULNESS,
            ]
        )
        assert settings.metrics == [MetricName.FAITHFULNESS, MetricName.CONTEXT_RECALL]

    def test_empty_metric_list_is_rejected(self):
        with pytest.raises(ValidationError, match="at least one metric"):
            EvaluationSettings(metrics=[])

    def test_backoff_ceiling_must_not_be_below_the_initial_delay(self):
        with pytest.raises(ValidationError, match="max_backoff_seconds"):
            LLMSettings(initial_backoff_seconds=10.0, max_backoff_seconds=1.0)

    def test_kms_encryption_requires_a_key(self):
        from llm_eval_guardrails.config import AWSSettings

        with pytest.raises(ValidationError, match="kms_key_id"):
            AWSSettings(server_side_encryption="aws:kms")

    def test_invalid_log_level_is_rejected(self):
        with pytest.raises(ValidationError, match="log_level"):
            Settings(log_level="CHATTY")

    def test_log_level_is_normalised_to_upper_case(self):
        assert Settings(log_level="debug").log_level == "DEBUG"

    def test_redaction_template_must_be_formattable(self):
        with pytest.raises(ValidationError, match="entity_type"):
            GuardrailSettings(pii_redaction_token="[{unknown_field}]")

    def test_empty_redaction_template_is_rejected(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            GuardrailSettings(pii_redaction_token="   ")

    def test_threshold_lookup(self):
        settings = EvaluationSettings(thresholds={MetricName.FAITHFULNESS: 0.9})
        assert settings.threshold_for(MetricName.FAITHFULNESS) == 0.9
        assert settings.threshold_for(MetricName.CONSISTENCY) is None

    def test_describe_masks_the_api_key(self):
        settings = Settings(llm=LLMSettings(api_key="super-secret-value"))
        described = settings.describe()
        assert described["llm"]["api_key"] == "***"
        assert "super-secret-value" not in str(described)

    def test_environment_variables_override_defaults(self, monkeypatch):
        monkeypatch.setenv("LEG_LLM__MODEL_ID", "env-model")
        monkeypatch.setenv("LEG_EVALUATION__MAX_CONCURRENCY", "32")
        monkeypatch.setenv("LEG_GUARDRAILS__BLOCK_ON_PII", "true")
        reset_settings_cache()
        try:
            settings = get_settings()
            assert settings.llm.model_id == "env-model"
            assert settings.evaluation.max_concurrency == 32
            assert settings.guardrails.block_on_pii is True
        finally:
            reset_settings_cache()

    def test_settings_are_cached(self):
        reset_settings_cache()
        try:
            assert get_settings() is get_settings()
        finally:
            reset_settings_cache()

    def test_invalid_environment_raises_configuration_error(self, monkeypatch):
        monkeypatch.setenv("LEG_LLM__TEMPERATURE", "99")
        reset_settings_cache()
        try:
            with pytest.raises(ConfigurationError, match="invalid configuration"):
                get_settings()
        finally:
            reset_settings_cache()
