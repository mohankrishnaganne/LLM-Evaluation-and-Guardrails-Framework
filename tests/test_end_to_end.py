"""End-to-end tests driving the CLI over local files with a stubbed judge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval_guardrails.config import reset_settings_cache
from llm_eval_guardrails.llm.registry import register_provider
from llm_eval_guardrails.workflows.run_evaluation import ExitCode, main

from .conftest import FakeLLMClient, claims_response

GROUNDED = claims_response("supported", "supported")
UNGROUNDED = claims_response("unsupported", "unsupported")


@pytest.fixture
def dataset(tmp_path: Path) -> str:
    path = tmp_path / "golden.jsonl"
    rows = [
        {
            "sample_id": f"s-{i}",
            "question": "What is the refund window?",
            "answer": "Refunds are accepted within 30 days.",
            "contexts": ["Refunds are accepted within 30 days of purchase."],
            "ground_truth": "30 days.",
        }
        for i in range(3)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def stub_judge(monkeypatch: pytest.MonkeyPatch):
    """Register a scripted judge and point configuration at it."""

    def _install(response: str) -> None:
        register_provider("stub", lambda _settings: FakeLLMClient([response]))
        monkeypatch.setenv("LEG_LLM__PROVIDER", "stub")
        monkeypatch.setenv("LEG_LLM__MAX_ATTEMPTS", "1")
        monkeypatch.setenv("LEG_EVALUATION__METRICS", '["faithfulness"]')
        monkeypatch.setenv("LEG_GUARDRAILS__USE_LLM_FOR_OUTBOUND", "false")
        reset_settings_cache()

    yield _install
    reset_settings_cache()


class TestCLIEndToEnd:
    def test_passing_run_exits_zero_and_writes_artifacts(self, dataset, tmp_path, stub_judge):
        stub_judge(GROUNDED)
        output = tmp_path / "reports"
        code = main(["--dataset", dataset, "--output", str(output), "--run-id", "run-1"])
        assert code == ExitCode.SUCCESS

        run_dir = output / "run-1"
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "summary.csv").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "detail-default.json").exists()

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["run_id"] == "run-1"
        assert summary["total_samples"] == 3
        assert summary["reports"][0]["aggregates"]["faithfulness"]["mean"] == 1.0

    def test_failing_quality_gate_exits_one(self, dataset, tmp_path, stub_judge):
        stub_judge(UNGROUNDED)
        code = main(
            [
                "--dataset",
                dataset,
                "--output",
                str(tmp_path / "reports"),
                "--run-id",
                "run-2",
                "--fail-under",
                "0.8",
            ]
        )
        assert code == ExitCode.QUALITY_GATE_FAILED

    def test_missing_dataset_exits_two(self, tmp_path, stub_judge):
        stub_judge(GROUNDED)
        code = main(
            [
                "--dataset",
                str(tmp_path / "nope.jsonl"),
                "--output",
                str(tmp_path / "reports"),
            ]
        )
        assert code == ExitCode.EXECUTION_ERROR

    def test_no_dataset_argument_exits_two(self, tmp_path, stub_judge):
        stub_judge(GROUNDED)
        assert main(["--output", str(tmp_path / "reports")]) == ExitCode.EXECUTION_ERROR

    def test_unknown_metric_exits_two(self, dataset, tmp_path, stub_judge):
        stub_judge(GROUNDED)
        code = main(["--dataset", dataset, "--output", str(tmp_path), "--metrics", "not_a_metric"])
        assert code == ExitCode.EXECUTION_ERROR

    def test_dry_run_validates_without_scoring(self, dataset, tmp_path, stub_judge):
        stub_judge(GROUNDED)
        output = tmp_path / "reports"
        code = main(["--dataset", dataset, "--output", str(output), "--dry-run", "--run-id", "r"])
        assert code == ExitCode.SUCCESS
        assert not output.exists()

    def test_limit_restricts_the_sample_count(self, dataset, tmp_path, stub_judge):
        stub_judge(GROUNDED)
        output = tmp_path / "reports"
        main(
            [
                "--dataset",
                dataset,
                "--output",
                str(output),
                "--run-id",
                "run-3",
                "--limit",
                "1",
            ]
        )
        summary = json.loads((output / "run-3" / "summary.json").read_text(encoding="utf-8"))
        assert summary["total_samples"] == 1

    def test_sweep_over_multiple_configurations(self, dataset, tmp_path, stub_judge):
        stub_judge(GROUNDED)
        configs = tmp_path / "configs.json"
        configs.write_text(
            json.dumps(
                [
                    {"name": "top_k_3", "params": {"top_k": 3}, "dataset_uri": dataset},
                    {"name": "top_k_10", "params": {"top_k": 10}, "dataset_uri": dataset},
                ]
            ),
            encoding="utf-8",
        )
        output = tmp_path / "reports"
        code = main(["--configs", str(configs), "--output", str(output), "--run-id", "run-4"])
        assert code == ExitCode.SUCCESS

        summary = json.loads((output / "run-4" / "summary.json").read_text(encoding="utf-8"))
        assert summary["config_count"] == 2
        assert {r["config_name"] for r in summary["reports"]} == {"top_k_3", "top_k_10"}
        assert (output / "run-4" / "detail-top_k_3.json").exists()

        markdown = (output / "run-4" / "report.md").read_text(encoding="utf-8")
        assert "| top_k_3 |" in markdown
        assert "| top_k_10 |" in markdown

    def test_guardrails_can_be_disabled(self, dataset, tmp_path, stub_judge):
        stub_judge(GROUNDED)
        output = tmp_path / "reports"
        code = main(
            [
                "--dataset",
                dataset,
                "--output",
                str(output),
                "--run-id",
                "run-5",
                "--no-guardrails",
            ]
        )
        assert code == ExitCode.SUCCESS
        summary = json.loads((output / "run-5" / "summary.json").read_text(encoding="utf-8"))
        assert summary["total_blocked_samples"] == 0

    def test_published_summary_can_be_read_back_into_the_model(self, dataset, tmp_path, stub_judge):
        from llm_eval_guardrails.models import EvaluationReport

        stub_judge(GROUNDED)
        output = tmp_path / "reports"
        main(["--dataset", dataset, "--output", str(output), "--run-id", "run-6"])
        detail = json.loads((output / "run-6" / "detail-default.json").read_text(encoding="utf-8"))
        # A regression gate must be able to load a previously published report.
        restored = EvaluationReport.model_validate(detail)
        assert restored.run_id == "run-6"
        assert restored.sample_count == 3
