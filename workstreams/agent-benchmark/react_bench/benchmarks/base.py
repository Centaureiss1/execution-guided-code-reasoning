from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from react_bench.config import RunnerConfig
from react_bench.protocol.schema import Observation, Task


class BenchmarkAdapter(ABC):
    benchmark: str

    @abstractmethod
    def load_tasks(self) -> list[Task]:
        raise NotImplementedError

    @abstractmethod
    def judge(self, task: Task, code: str, config: RunnerConfig) -> Observation:
        raise NotImplementedError


def load_jsonl_tasks(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
