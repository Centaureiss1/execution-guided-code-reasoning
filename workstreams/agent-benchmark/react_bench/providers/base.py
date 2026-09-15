from __future__ import annotations

from abc import ABC, abstractmethod

from react_bench.config import RunnerConfig
from react_bench.protocol.schema import GenerationResult


class BaseProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        config: RunnerConfig,
        is_recovery: bool = False,
    ) -> GenerationResult:
        raise NotImplementedError
