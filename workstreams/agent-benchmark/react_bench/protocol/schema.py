from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Task:
    benchmark: str
    dataset: str
    task_id: str
    language: str
    prompt: str
    tests_metadata: dict[str, Any] = field(default_factory=dict)
    scenario: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(**data)


@dataclass
class Observation:
    status: str
    error_type: str | None
    obs_for_model: str
    obs_full: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        return cls(**data)


@dataclass
class GenerationResult:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency: float | None = None


@dataclass
class Turn:
    turn_index: int
    raw_output: str
    parsed_think: str
    parsed_submit: str
    parse_status: str
    obs_for_model: str
    obs_full: dict[str, Any]
    token_usage: dict[str, int | None] = field(default_factory=dict)
    latency: float | None = None
    finish_reason: str | None = None
    recovery_used: bool = False
    recovery_raw_output: str = ""
    recovery_parse_status: str | None = None
    recovery_finish_reason: str | None = None
    recovery_token_usage: dict[str, int | None] = field(default_factory=dict)
    recovery_disable_thinking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        return cls(**data)


@dataclass
class Trace:
    task: Task
    provider: str
    model: str
    config: dict[str, Any]
    turns: list[Turn] = field(default_factory=list)
    final_status: str = "running"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def append_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        if turn.obs_full.get("status") in {"pass", "fail"}:
            self.final_status = turn.obs_full["status"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "task": self.task.to_dict(),
            "provider": self.provider,
            "model": self.model,
            "config": self.config,
            "turns": [turn.to_dict() for turn in self.turns],
            "final_status": self.final_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        trace = cls(
            run_id=data["run_id"],
            created_at=data.get("created_at", time.time()),
            task=Task.from_dict(data["task"]),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            config=data.get("config", {}),
            turns=[Turn.from_dict(turn) for turn in data.get("turns", [])],
            final_status=data.get("final_status", "running"),
        )
        return trace

    def to_tagged_text(self) -> str:
        parts = [f"<input>{self.task.prompt}</input>"]
        for turn in self.turns:
            if turn.parsed_think:
                parts.append(f"<think>{turn.parsed_think}</think>")
            if turn.parsed_submit:
                parts.append(f"<submit>{turn.parsed_submit}</submit>")
            if turn.obs_for_model:
                parts.append(f"<obs>{turn.obs_for_model}</obs>")
        return "\n".join(parts)
