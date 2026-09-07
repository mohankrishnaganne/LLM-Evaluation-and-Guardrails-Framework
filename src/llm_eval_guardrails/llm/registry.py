"""Factory for constructing evaluator-LLM clients from configuration.

Provider modules are imported lazily so that a deployment installing only the
Bedrock dependencies never imports the OpenAI SDK, and vice versa. Additional
providers can be registered at runtime via :func:`register_provider`, which
keeps the framework open for extension without editing this module.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import LLMProvider, Settings
from ..exceptions import ConfigurationError
from ..logging_config import get_logger
from .base import LLMClient

__all__ = ["build_llm_client", "register_provider"]

_log = get_logger(__name__)

#: Custom provider factories keyed by provider name.
_CUSTOM: dict[str, Callable[[Settings], LLMClient]] = {}


def register_provider(name: str, factory: Callable[[Settings], LLMClient]) -> None:
    """Register a custom evaluator-LLM provider.

    Args:
        name: Provider identifier, matched case-insensitively against
            ``settings.llm.provider``.
        factory: Callable that builds a client from the full settings object.

    Raises:
        ValueError: If ``name`` is empty.
    """
    key = name.strip().lower()
    if not key:
        msg = "provider name must not be empty"
        raise ValueError(msg)
    _CUSTOM[key] = factory
    _log.info("llm.provider_registered", provider=key)


def build_llm_client(settings: Settings) -> LLMClient:
    """Construct the evaluator-LLM client described by ``settings``.

    Args:
        settings: The fully-resolved application settings.

    Returns:
        A ready-to-use client for the configured provider.

    Raises:
        ConfigurationError: If the provider is unknown or its dependencies are
            not installed.
    """
    provider = settings.llm.provider
    key = provider.value if isinstance(provider, LLMProvider) else str(provider).lower()

    custom = _CUSTOM.get(key)
    if custom is not None:
        return custom(settings)

    if key == LLMProvider.BEDROCK.value:
        from .bedrock import BedrockLLMClient

        client: LLMClient = BedrockLLMClient(settings.llm, settings.aws)
    elif key == LLMProvider.OPENAI.value:
        from .openai_client import OpenAILLMClient

        client = OpenAILLMClient(settings.llm)
    else:  # pragma: no cover - unreachable while LLMProvider is exhaustive
        known = sorted({p.value for p in LLMProvider} | set(_CUSTOM))
        msg = f"unknown LLM provider {key!r}; known providers: {known}"
        raise ConfigurationError(msg)

    _log.info("llm.client_built", provider=key, model_id=settings.llm.model_id)
    return client
