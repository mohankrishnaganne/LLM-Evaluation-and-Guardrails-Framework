"""Tests for S3 storage, dataset loading, reporting and the CLI workflow."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_eval_guardrails.aws.dataset import DatasetLoader, normalize_record
from llm_eval_guardrails.aws.s3 import S3Storage, is_s3_uri, parse_s3_uri
from llm_eval_guardrails.config import AWSSettings, EvaluationSettings, Settings
from llm_eval_guardrails.exceptions import DatasetError, StorageError
from llm_eval_guardrails.models import AggregateMetric, EvaluationReport, MetricName
from llm_eval_guardrails.workflows.reporting import render_markdown, summarize_sweep
from llm_eval_guardrails.workflows.run_evaluation import (
    ExitCode,
    build_parser,
    evaluate_gates,
    load_configs,
)


@pytest.fixture
def storage() -> S3Storage:
    return S3Storage(AWSSettings(report_bucket="test-bucket"))


class TestS3UriParsing:
    def test_recognises_s3_uris(self):
        assert is_s3_uri("s3://bucket/key")
        assert not is_s3_uri("/local/path")
        assert not is_s3_uri("https://example.com/x")

    def test_parses_bucket_and_key(self):
        location = parse_s3_uri("s3://my-bucket/a/b/c.jsonl")
        assert location.bucket == "my-bucket"
        assert location.key == "a/b/c.jsonl"
        assert location.uri == "s3://my-bucket/a/b/c.jsonl"

    @pytest.mark.parametrize(
        "uri", ["s3://bucket", "s3://bucket/", "s3://", "/local/path", "s3:///key"]
    )
    def test_rejects_malformed_uris(self, uri):
        with pytest.raises(StorageError):
            parse_s3_uri(uri)


class TestLocalStorage:
    def test_round_trips_text(self, storage, tmp_path):
        path = str(tmp_path / "out.txt")
        storage.write_text(path, "hello")
        assert storage.read_text(path) == "hello"

    def test_creates_parent_directories(self, storage, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "out.txt")
        storage.write_text(path, "hi")
        assert Path(path).read_text(encoding="utf-8") == "hi"

    def test_missing_file_raises_storage_error(self, storage, tmp_path):
        with pytest.raises(StorageError, match="failed to read"):
            storage.read_text(str(tmp_path / "absent.txt"))

    def test_round_trips_json(self, storage, tmp_path):
        path = str(tmp_path / "out.json")
        storage.write_json(path, {"a": [1, 2], "b": None})
        assert json.loads(Path(path).read_text(encoding="utf-8")) == {"a": [1, 2], "b": None}

    def test_reads_jsonl(self, storage, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
        assert storage.read_jsonl(str(path)) == [{"a": 1}, {"a": 2}]

    def test_jsonl_error_reports_the_line_number(self, storage, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(DatasetError, match=":2:"):
            storage.read_jsonl(str(path))

    def test_jsonl_rejects_non_object_lines(self, storage, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="expected a JSON object"):
            storage.read_jsonl(str(path))

    def test_reads_csv(self, storage, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        assert storage.read_csv(str(path)) == [{"a": "1", "b": "2"}]

    def test_writes_csv_with_inferred_columns(self, storage, tmp_path):
        path = str(tmp_path / "out.csv")
        storage.write_csv(path, [{"a": 1, "b": 2}, {"a": 3, "c": 4}])
        rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines()))
        assert rows[0]["a"] == "1"
        assert "c" in rows[0]

    def test_writing_csv_with_no_columns_is_an_error(self, storage, tmp_path):
        with pytest.raises(DatasetError, match="no columns"):
            storage.write_csv(str(tmp_path / "out.csv"), [])

    def test_report_uri_uses_configured_bucket_and_prefix(self, storage):
        uri = storage.report_uri("run-1", "cfg name", "report.json")
        assert uri.startswith("s3://test-bucket/llm-eval/reports/run-1/")
        assert " " not in uri

    def test_report_uri_requires_a_bucket(self):
        with pytest.raises(StorageError, match="no report bucket"):
            S3Storage(AWSSettings()).report_uri("r", "c", "report.json")


class TestS3Interaction:
    class FakeS3:
        def __init__(self):
            self.objects: dict[tuple[str, str], bytes] = {}
            self.put_requests: list[dict] = []

        def get_object(self, Bucket, Key) -> dict:  # noqa: N803 - boto3 casing
            from botocore.exceptions import ClientError

            payload = self.objects.get((Bucket, Key))
            if payload is None:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
                )

            body = payload

            class Body:
                @staticmethod
                def read() -> bytes:
                    return body

            return {"Body": Body(), "ContentLength": len(body)}

        def put_object(self, **request):
            self.put_requests.append(request)
            self.objects[(request["Bucket"], request["Key"])] = request["Body"]
            return {}

    def test_reads_an_object(self):
        fake = self.FakeS3()
        fake.objects[("b", "k")] = b"payload"
        storage = S3Storage(AWSSettings(), client=fake)
        assert storage.read_text("s3://b/k") == "payload"

    def test_missing_key_raises_storage_error(self):
        storage = S3Storage(AWSSettings(), client=self.FakeS3())
        with pytest.raises(StorageError, match="NoSuchKey"):
            storage.read_text("s3://b/absent")

    def test_applies_default_sse(self):
        fake = self.FakeS3()
        storage = S3Storage(AWSSettings(), client=fake)
        storage.write_text("s3://b/k", "x")
        assert fake.put_requests[0]["ServerSideEncryption"] == "AES256"

    def test_applies_kms_key_when_configured(self):
        fake = self.FakeS3()
        settings = AWSSettings(server_side_encryption="aws:kms", kms_key_id="arn:key")
        S3Storage(settings, client=fake).write_text("s3://b/k", "x")
        assert fake.put_requests[0]["SSEKMSKeyId"] == "arn:key"

    def test_no_sse_when_disabled(self):
        fake = self.FakeS3()
        settings = AWSSettings(server_side_encryption=None)
        S3Storage(settings, client=fake).write_text("s3://b/k", "x")
        assert "ServerSideEncryption" not in fake.put_requests[0]

    async def test_async_wrappers_work(self):
        fake = self.FakeS3()
        storage = S3Storage(AWSSettings(), client=fake)
        await storage.awrite_text("s3://b/k", "async payload")
        assert await storage.aread_text("s3://b/k") == "async payload"


class TestNormalizeRecord:
    def test_normalises_canonical_fields(self):
        sample = normalize_record(
            {
                "sample_id": "x",
                "question": "q?",
                "answer": "a",
                "contexts": ["c1", "c2"],
                "ground_truth": "gt",
            },
            0,
        )
        assert sample.sample_id == "x"
        assert len(sample.contexts) == 2
        assert sample.ground_truth == "gt"

    @pytest.mark.parametrize("alias", ["question", "query", "input", "prompt", "user_input"])
    def test_accepts_question_aliases(self, alias):
        assert normalize_record({alias: "q?"}, 0).question == "q?"

    @pytest.mark.parametrize("alias", ["answer", "response", "output", "prediction"])
    def test_accepts_answer_aliases(self, alias):
        assert normalize_record({"question": "q?", alias: "a"}, 0).answer == "a"

    def test_synthesises_a_sample_id_when_missing(self):
        assert normalize_record({"question": "q?"}, 7).sample_id == "sample-7"

    def test_missing_question_is_an_error(self):
        with pytest.raises(DatasetError, match="no usable question"):
            normalize_record({"answer": "a"}, 0)

    def test_blank_question_is_an_error(self):
        with pytest.raises(DatasetError):
            normalize_record({"question": "   "}, 0)

    def test_contexts_as_objects(self):
        sample = normalize_record(
            {
                "question": "q?",
                "contexts": [{"doc_id": "d1", "content": "text", "score": 0.9}],
            },
            0,
        )
        assert sample.contexts[0].doc_id == "d1"
        assert sample.contexts[0].score == 0.9

    @pytest.mark.parametrize("key", ["content", "text", "page_content", "chunk", "passage"])
    def test_context_text_aliases(self, key):
        sample = normalize_record({"question": "q?", "contexts": [{key: "text"}]}, 0)
        assert sample.contexts[0].content == "text"

    def test_contexts_as_json_encoded_string(self):
        sample = normalize_record({"question": "q?", "contexts": '["a", "b"]'}, 0)
        assert sample.context_texts == ["a", "b"]

    def test_contexts_as_delimited_string(self):
        sample = normalize_record({"question": "q?", "contexts": "a|||b"}, 0)
        assert sample.context_texts == ["a", "b"]

    def test_contexts_as_plain_string(self):
        sample = normalize_record({"question": "q?", "contexts": "single chunk"}, 0)
        assert sample.context_texts == ["single chunk"]

    def test_empty_contexts_are_dropped(self):
        sample = normalize_record({"question": "q?", "contexts": ["a", "", "  "]}, 0)
        assert sample.context_texts == ["a"]

    def test_ranks_are_assigned_in_order(self):
        sample = normalize_record({"question": "q?", "contexts": ["a", "b"]}, 0)
        assert [c.rank for c in sample.contexts] == [0, 1]

    def test_ground_truth_list_takes_the_first_entry(self):
        sample = normalize_record({"question": "q?", "ground_truths": ["first", "second"]}, 0)
        assert sample.ground_truth == "first"

    def test_empty_ground_truth_becomes_none(self):
        assert normalize_record({"question": "q?", "ground_truth": "  "}, 0).ground_truth is None

    def test_metadata_from_json_string(self):
        sample = normalize_record({"question": "q?", "metadata": '{"tenant": "acme"}'}, 0)
        assert sample.metadata == {"tenant": "acme"}

    def test_unparseable_metadata_becomes_empty(self):
        assert normalize_record({"question": "q?", "metadata": "not json"}, 0).metadata == {}


class TestDatasetLoader:
    def test_loads_jsonl(self, jsonl_dataset):
        samples = DatasetLoader(AWSSettings()).load(jsonl_dataset)
        assert len(samples) == 3
        assert samples[0].sample_id == "s-0"

    def test_limit_caps_the_sample_count(self, jsonl_dataset):
        assert len(DatasetLoader(AWSSettings()).load(jsonl_dataset, limit=2)) == 2

    def test_loads_csv(self, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("question,answer,contexts\nq1?,a1,ctx1|||ctx2\n", encoding="utf-8")
        samples = DatasetLoader(AWSSettings()).load(str(path))
        assert samples[0].context_texts == ["ctx1", "ctx2"]

    def test_malformed_rows_are_skipped_by_default(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"question": "ok?"}\n{"answer": "no question"}\n', encoding="utf-8")
        assert len(DatasetLoader(AWSSettings()).load(str(path))) == 1

    def test_strict_mode_fails_on_a_bad_row(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"question": "ok?"}\n{"answer": "no question"}\n', encoding="utf-8")
        with pytest.raises(DatasetError):
            DatasetLoader(AWSSettings()).load(str(path), strict=True)

    def test_a_dataset_with_no_valid_samples_is_an_error(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"answer": "no question"}\n', encoding="utf-8")
        with pytest.raises(DatasetError, match="no valid samples"):
            DatasetLoader(AWSSettings()).load(str(path))

    async def test_async_load(self, jsonl_dataset):
        assert len(await DatasetLoader(AWSSettings()).aload(jsonl_dataset)) == 3


def make_report(name: str, mean: float, **kwargs) -> EvaluationReport:
    start = datetime.now(timezone.utc)
    defaults = {
        "run_id": "r1",
        "config_name": name,
        "started_at": start,
        "completed_at": start + timedelta(seconds=1),
        "sample_count": 10,
        "aggregates": {
            MetricName.FAITHFULNESS: AggregateMetric(
                metric=MetricName.FAITHFULNESS,
                count=10,
                mean=mean,
                pass_rate=mean,
                threshold=0.8,
            )
        },
    }
    defaults.update(kwargs)
    return EvaluationReport(**defaults)


class TestReporting:
    def test_identifies_the_best_configuration(self):
        summary = summarize_sweep(
            "r1", [make_report("a", 0.7), make_report("b", 0.95), make_report("c", 0.8)]
        )
        assert summary.best_config(MetricName.FAITHFULNESS) == "b"

    def test_best_config_is_none_without_results(self):
        summary = summarize_sweep("r1", [])
        assert summary.best_config(MetricName.FAITHFULNESS) is None

    def test_failure_rate_is_sample_weighted(self):
        summary = summarize_sweep(
            "r1",
            [
                make_report("a", 0.9, sample_count=90, failure_rate=0.0),
                make_report("b", 0.9, sample_count=10, failure_rate=1.0),
            ],
        )
        assert summary.total_failure_rate() == pytest.approx(0.1)

    def test_failure_rate_of_an_empty_sweep_is_zero(self):
        assert summarize_sweep("r1", []).total_failure_rate() == 0.0

    def test_to_dict_omits_samples_by_default(self, sample):
        from llm_eval_guardrails.models import SampleResult

        report = make_report("a", 0.9, sample_results=[SampleResult(sample_id="s")])
        payload = summarize_sweep("r1", [report]).to_dict()
        assert payload["reports"][0]["sample_results"] == []

    def test_to_dict_can_include_samples(self):
        from llm_eval_guardrails.models import SampleResult

        report = make_report("a", 0.9, sample_results=[SampleResult(sample_id="s")])
        payload = summarize_sweep("r1", [report]).to_dict(include_samples=True)
        assert len(payload["reports"][0]["sample_results"]) == 1

    def test_guardrail_triggers_are_merged_across_configs(self):
        summary = summarize_sweep(
            "r1",
            [
                make_report("a", 0.9, guardrail_triggers={"inbound:pii": 2}),
                make_report("b", 0.9, guardrail_triggers={"inbound:pii": 3}),
            ],
        )
        assert summary.to_dict()["guardrail_triggers"]["inbound:pii"] == 5

    def test_markdown_contains_a_row_per_config(self):
        markdown = render_markdown(
            summarize_sweep("r1", [make_report("alpha", 0.9), make_report("beta", 0.7)])
        )
        assert "| alpha |" in markdown
        assert "| beta |" in markdown
        assert "0.900" in markdown

    def test_markdown_handles_an_empty_sweep(self):
        assert "_No configurations were evaluated._" in render_markdown(summarize_sweep("r1", []))

    def test_markdown_reports_guardrail_triggers(self):
        markdown = render_markdown(
            summarize_sweep("r1", [make_report("a", 0.9, guardrail_triggers={"inbound:pii": 4})])
        )
        assert "`inbound:pii`" in markdown
        assert "| 4 |" in markdown


class TestQualityGates:
    def _settings(self, threshold: float = 0.8, max_failure_rate: float = 0.25) -> Settings:
        return Settings(
            evaluation=EvaluationSettings(
                metrics=[MetricName.FAITHFULNESS],
                thresholds={MetricName.FAITHFULNESS: threshold},
                max_failure_rate=max_failure_rate,
            )
        )

    def test_passing_run_has_no_failures(self):
        summary = summarize_sweep("r1", [make_report("a", 0.9)])
        assert evaluate_gates(summary, self._settings()) == []

    def test_metric_below_threshold_fails(self):
        summary = summarize_sweep("r1", [make_report("a", 0.5)])
        failures = evaluate_gates(summary, self._settings())
        assert len(failures) == 1
        assert "faithfulness" in failures[0]

    def test_mean_equal_to_threshold_passes(self):
        summary = summarize_sweep("r1", [make_report("a", 0.8)])
        assert evaluate_gates(summary, self._settings(threshold=0.8)) == []

    def test_excessive_failure_rate_fails(self):
        summary = summarize_sweep("r1", [make_report("a", 0.9, failure_rate=0.9)])
        failures = evaluate_gates(summary, self._settings())
        assert any("failure rate" in f for f in failures)

    def test_failures_are_reported_per_configuration(self):
        summary = summarize_sweep("r1", [make_report("a", 0.1), make_report("b", 0.2)])
        failures = evaluate_gates(summary, self._settings())
        assert len(failures) == 2


class TestCLIParsing:
    def test_parses_core_flags(self):
        args = build_parser().parse_args(
            ["--dataset", "s3://b/d.jsonl", "--output", "s3://b/out", "--limit", "5"]
        )
        assert args.dataset == "s3://b/d.jsonl"
        assert args.limit == 5

    def test_flags_default_to_false(self):
        args = build_parser().parse_args([])
        assert not args.dry_run
        assert not args.no_guardrails
        assert not args.strict_dataset

    def test_loads_a_config_list(self, storage, tmp_path):
        path = tmp_path / "configs.json"
        path.write_text(json.dumps([{"name": "a", "params": {"top_k": 3}}]), encoding="utf-8")
        configs = load_configs(str(path), storage)
        assert configs[0].name == "a"
        assert configs[0].params["top_k"] == 3

    def test_loads_a_wrapped_config_object(self, storage, tmp_path):
        path = tmp_path / "configs.json"
        path.write_text(json.dumps({"configs": [{"name": "a"}]}), encoding="utf-8")
        assert load_configs(str(path), storage)[0].name == "a"

    def test_invalid_json_is_a_configuration_error(self, storage, tmp_path):
        from llm_eval_guardrails.exceptions import ConfigurationError

        path = tmp_path / "configs.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not valid JSON"):
            load_configs(str(path), storage)

    def test_non_list_payload_is_a_configuration_error(self, storage, tmp_path):
        from llm_eval_guardrails.exceptions import ConfigurationError

        path = tmp_path / "configs.json"
        path.write_text(json.dumps({"unexpected": 1}), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="must contain a JSON list"):
            load_configs(str(path), storage)

    def test_invalid_config_entry_is_rejected(self, storage, tmp_path):
        from llm_eval_guardrails.exceptions import ConfigurationError

        path = tmp_path / "configs.json"
        path.write_text(json.dumps([{"no_name_field": 1}]), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="invalid RAG configuration"):
            load_configs(str(path), storage)


class TestExitCodes:
    def test_codes_are_distinct(self):
        assert ExitCode.SUCCESS == 0
        assert ExitCode.QUALITY_GATE_FAILED == 1
        assert ExitCode.EXECUTION_ERROR == 2
