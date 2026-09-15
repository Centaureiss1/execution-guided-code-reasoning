from __future__ import annotations

from react_bench.providers.base import BaseProvider
from react_bench.providers.openai_compatible import (
    AzureOpenAIProvider,
    OpenAICompatibleProvider,
    VLLMProvider,
)


def create_provider(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> BaseProvider:
    normalized = provider.lower()
    if normalized == "azure_openai":
        return AzureOpenAIProvider(model, base_url, api_key, api_key_env)
    if normalized == "vllm":
        return VLLMProvider(model, base_url, api_key, api_key_env)
    if normalized in {"openai_compatible", "openai"}:
        return OpenAICompatibleProvider(model, base_url, api_key, api_key_env)
    raise ValueError(f"Unsupported provider: {provider}")
