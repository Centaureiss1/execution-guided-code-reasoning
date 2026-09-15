from __future__ import annotations

import os
import time
from typing import Any

from react_bench.config import RunnerConfig
from react_bench.protocol.schema import GenerationResult
from react_bench.providers.base import BaseProvider


class OpenAICompatibleProvider(BaseProvider):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        name: str = "openai_compatible",
    ) -> None:
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv(api_key_env or "OPENAI_API_KEY") or "EMPTY"
        self.name = name

    def generate(
        self,
        messages: list[dict[str, str]],
        config: RunnerConfig,
        is_recovery: bool = False,
    ) -> GenerationResult:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for API providers. "
                "Install with: pip install 'react-bench[openai]' or pip install openai"
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        started = time.time()
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": _sampling_value(
                config.temperature,
                config.recovery_temperature,
                is_recovery,
            ),
            "max_tokens": config.max_output_tokens,
        }
        top_p = _sampling_value(config.top_p, config.recovery_top_p, is_recovery)
        presence_penalty = _sampling_value(
            config.presence_penalty,
            config.recovery_presence_penalty,
            is_recovery,
        )
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        extra_body = self._extra_body(config, is_recovery)
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**request_kwargs)
        latency = time.time() - started
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency=latency,
        )

    def _extra_body(self, config: RunnerConfig, is_recovery: bool) -> dict[str, Any]:
        return {}


class VLLMProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url or os.getenv("VLLM_BASE_URL") or "http://localhost:8000/v1",
            api_key=api_key,
            api_key_env=api_key_env or "VLLM_API_KEY",
            name="vllm",
        )

    def _extra_body(self, config: RunnerConfig, is_recovery: bool) -> dict[str, Any]:
        body: dict[str, Any] = {}
        top_k = _sampling_value(config.top_k, config.recovery_top_k, is_recovery)
        min_p = _sampling_value(config.min_p, config.recovery_min_p, is_recovery)
        if top_k is not None:
            body["top_k"] = top_k
        if min_p is not None:
            body["min_p"] = min_p
        if is_recovery and config.recovery_disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["enable_thinking"] = False
        return body


def _sampling_value(default: Any, recovery: Any, is_recovery: bool) -> Any:
    if is_recovery and recovery is not None:
        return recovery
    return default


class AzureOpenAIProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        endpoint = base_url or os.getenv("AZURE_OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
        if endpoint and not endpoint.rstrip("/").endswith("/openai/v1"):
            endpoint = endpoint.rstrip("/") + "/openai/v1/"
        super().__init__(
            model=model,
            base_url=endpoint,
            api_key=api_key,
            api_key_env=api_key_env or "AZURE_OPENAI_API_KEY",
            name="azure_openai",
        )
