from __future__ import annotations

from dataclasses import asdict

from react_bench.agent.runner import ReActRunner
from react_bench.benchmarks.factory import create_benchmark_adapter
from react_bench.config import RunnerConfig
from react_bench.logging.jsonl import read_jsonl
from react_bench.logging.run_logger import RunLogger
from react_bench.protocol.schema import Trace
from react_bench.providers.base import BaseProvider


def distill_failed_prefixes(
    failed_prefixes_path: str,
    teacher_provider: BaseProvider,
    config: RunnerConfig,
    logs_dir: str,
    max_teacher_turns: int | None = None,
) -> tuple[int, int]:
    logger = RunLogger(logs_dir)
    accepted = 0
    rejected = 0
    for row in read_jsonl(failed_prefixes_path):
        prefix = Trace.from_dict(row)
        adapter = create_benchmark_adapter(
            benchmark=prefix.task.benchmark,
            dataset=prefix.task.dataset,
            scenario=prefix.task.scenario,
        )
        teacher_config = config
        if max_teacher_turns is not None:
            teacher_config = RunnerConfig(**{**asdict(config), "max_turns": len(prefix.turns) + max_teacher_turns})
        prefix.provider = teacher_provider.name
        prefix.model = teacher_provider.model
        prefix.config = asdict(teacher_config)
        runner = ReActRunner(adapter, teacher_provider, teacher_config)
        trace = runner.run_task(prefix.task, prefix=prefix)
        if trace.final_status == "pass":
            logger.log_teacher_trace(trace)
            accepted += 1
        else:
            logger.log_rejected_trace(trace)
            rejected += 1
    return accepted, rejected
