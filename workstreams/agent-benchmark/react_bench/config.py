from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class RunnerConfig:
    context_window: int = 32768
    prompt_budget_tokens: int = 3000
    lcb_long_prompt_budget_tokens: int = 6000
    lcb_long_prompt_threshold_tokens: int = 2400
    max_output_tokens: int = 16000
    safety_margin_tokens: int = 1024
    max_turns: int = 3
    temperature: float = 0.6
    top_p: float | None = 0.95
    top_k: int | None = 20
    min_p: float | None = 0.0
    presence_penalty: float | None = None
    recovery_temperature: float | None = 0.7
    recovery_top_p: float | None = 0.8
    recovery_top_k: int | None = 20
    recovery_min_p: float | None = 0.0
    recovery_presence_penalty: float | None = None
    feedback_mode: str = "public"
    trace_compaction: bool = True
    stop_on_pass: bool = True
    timeout_seconds: float = 10.0
    stderr_stdout_limit: int = 4000
    logs_dir: str = "logs"
    enable_submit_recovery: bool = True
    recovery_max_output_tokens: int = 4096
    length_failure_hint: bool = True
    recovery_reasoning_tail_chars: int = 2000
    recovery_disable_thinking: bool = True
    stop_after_recovery_failure: bool = True

    @property
    def max_prompt_tokens(self) -> int:
        return max(0, self.context_window - self.max_output_tokens - self.safety_margin_tokens)

    def effective_prompt_budget(self, benchmark: str, prompt_tokens: int) -> int:
        configured = min(self.prompt_budget_tokens, self.max_prompt_tokens)
        if (
            benchmark == "livecodebench"
            and prompt_tokens >= self.lcb_long_prompt_threshold_tokens
        ):
            return min(self.lcb_long_prompt_budget_tokens, self.max_prompt_tokens)
        return configured


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> RunnerConfig:
    data: dict[str, Any] = {}
    if path:
        data.update(_load_mapping(Path(path)))
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    allowed = {field.name for field in fields(RunnerConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
    return RunnerConfig(**data)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        return tomllib.loads(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Missing config key in line: {raw_line!r}")
        data[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value
