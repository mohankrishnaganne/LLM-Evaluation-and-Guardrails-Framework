"""Enterprise LLM evaluation and guardrails framework for RAG applications.

The framework has three layers:

* **Evaluation** -- :class:`~llm_eval_guardrails.evaluators.suite.EvaluationSuite`
  scores retrieval relevance, response grounding, answer quality and answer
  stability using a pluggable LLM judge.
* **Guardrails** -- :class:`~llm_eval_guardrails.guardrails.pipeline.GuardrailPipeline`
  enforces inbound prompt policy (PII, injection) and outbound response policy
  (unsupported claims, leakage).
* **Workflows** -- batch ingestion from S3, multi-configuration sweeps and
  report publication.

Quick start::

    from llm_eval_guardrails import GuardrailPipeline, get_settings

    settings = get_settings()
    pipeline = GuardrailPipeline.from_settings(settings)
    report = pipeline.check_prompt("What is our refund policy?")
"""

from .config import Settings, get_settings, reset_settings_cache
from .evaluators.suite import EvaluationSuite
from .exceptions import (
    ConfigurationError,
    DatasetError,
    EvaluationError,
    FrameworkError,
    GuardrailTripped,
    LLMError,
    StorageError,
)
from .guardrails.pipeline import GuardrailPipeline
from .llm.registry import build_llm_client
from .logging_config import configure_logging, get_logger
from .models import (
    EvaluationReport,
    GuardrailAction,
    GuardrailReport,
    MetricName,
    RAGSample,
    RetrievedDocument,
    RunConfig,
    SampleResult,
    Severity,
)

__version__ = "0.1.0"

__all__ = [
    "ConfigurationError",
    "DatasetError",
    "EvaluationError",
    "EvaluationReport",
    "EvaluationSuite",
    "FrameworkError",
    "GuardrailAction",
    "GuardrailPipeline",
    "GuardrailReport",
    "GuardrailTripped",
    "LLMError",
    "MetricName",
    "RAGSample",
    "RetrievedDocument",
    "RunConfig",
    "SampleResult",
    "Settings",
    "Severity",
    "StorageError",
    "__version__",
    "build_llm_client",
    "configure_logging",
    "get_logger",
    "get_settings",
    "reset_settings_cache",
]
