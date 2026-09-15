from __future__ import annotations

from react_bench.benchmarks.base import BenchmarkAdapter
from react_bench.benchmarks.evalplus import EvalPlusAdapter
from react_bench.benchmarks.kodcode import KodCodeAdapter
from react_bench.benchmarks.livecodebench import LiveCodeBenchAdapter


def create_benchmark_adapter(
    benchmark: str,
    dataset: str | None = None,
    scenario: str | None = None,
    tasks_path: str | None = None,
    limit: int | None = None,
) -> BenchmarkAdapter:
    normalized = benchmark.lower()
    if normalized == "evalplus":
        return EvalPlusAdapter(dataset or "humaneval", tasks_path, limit)
    if normalized == "livecodebench":
        return LiveCodeBenchAdapter(dataset or "release_v1", scenario or "codegeneration", tasks_path, limit)
    if normalized == "kodcode":
        return KodCodeAdapter(dataset or "train", tasks_path, limit)
    raise ValueError(f"Unsupported benchmark: {benchmark}")
