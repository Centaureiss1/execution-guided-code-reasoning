from __future__ import annotations

import json
from typing import Any

from react_bench.benchmarks.base import BenchmarkAdapter, load_jsonl_tasks
from react_bench.config import RunnerConfig
from react_bench.execution import PythonExecutor
from react_bench.protocol.schema import Observation, Task


class LiveCodeBenchAdapter(BenchmarkAdapter):
    benchmark = "livecodebench"

    def __init__(
        self,
        dataset: str = "release_v1",
        scenario: str = "codegeneration",
        tasks_path: str | None = None,
        limit: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.scenario = scenario
        self.tasks_path = tasks_path
        self.limit = limit

    def load_tasks(self) -> list[Task]:
        if not self.tasks_path:
            raise RuntimeError(
                "LiveCodeBench official package loading is not bundled in v1. "
                "Export LCB tasks to JSONL and pass --tasks. Expected fields include "
                "task_id/question_id, prompt/question_content, and public_test_cases/test_cases."
            )
        tasks = [self._task_from_row(row) for row in load_jsonl_tasks(self.tasks_path)]
        return tasks[: self.limit] if self.limit else tasks

    def judge(self, task: Task, code: str, config: RunnerConfig) -> Observation:
        executor = PythonExecutor(config.timeout_seconds, config.stderr_stdout_limit)
        cases = self._select_test_cases(task.tests_metadata, config.feedback_mode)
        functional_cases = [case for case in cases if _is_functional_case(case)]
        stdio_cases = [case for case in cases if not _is_functional_case(case)]
        if functional_cases and not stdio_cases:
            metadata = task.tests_metadata.get("metadata") or {}
            func_name = metadata.get("func_name")
            return executor.run_lcb_functional_tests(code, str(func_name or ""), functional_cases)
        return executor.run_stdio_tests(code, stdio_cases or cases)

    def _task_from_row(self, row: dict[str, Any]) -> Task:
        task_id = str(row.get("task_id") or row.get("question_id") or row.get("id"))
        prompt = str(row.get("prompt") or row.get("question_content") or row.get("question") or "")
        metadata = {
            "public_test_cases": coerce_lcb_cases(row.get("public_test_cases") or row.get("sample_tests")),
            "private_test_cases": coerce_lcb_cases(row.get("private_test_cases") or row.get("hidden_tests")),
            "test_cases": coerce_lcb_cases(row.get("test_cases") or row.get("tests")),
            "metadata": row.get("metadata") or {},
        }
        prompt = _append_public_examples(prompt, metadata["public_test_cases"])
        return Task(
            benchmark=self.benchmark,
            dataset=self.dataset,
            scenario=self.scenario,
            task_id=task_id,
            language="python",
            prompt=prompt,
            tests_metadata=metadata,
        )

    def _select_test_cases(self, metadata: dict[str, Any], feedback_mode: str) -> list[dict[str, Any]]:
        public = list(metadata.get("public_test_cases") or [])
        private = list(metadata.get("private_test_cases") or [])
        all_cases = list(metadata.get("test_cases") or [])
        if feedback_mode in {"oracle", "hidden"}:
            return all_cases or public + private
        return public or all_cases


def coerce_lcb_cases(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, list):
        cases = []
        for item in value:
            if isinstance(item, dict):
                cases.append(
                    {
                        "input": item.get("input", ""),
                        "output": item.get("output", ""),
                        "testtype": item.get("testtype") or item.get("test_type") or item.get("type"),
                    }
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                cases.append({"input": item[0], "output": item[1]})
        return cases
    return []


def _append_public_examples(prompt: str, public_cases: list[dict[str, Any]], limit: int = 3) -> str:
    if not public_cases:
        return prompt
    parts = [prompt.rstrip(), "\nPublic examples:"]
    for index, case in enumerate(public_cases[:limit], start=1):
        parts.append(
            "\n".join(
                [
                    f"Example {index} input:",
                    str(case.get("input", "")).rstrip("\n"),
                    f"Example {index} output:",
                    str(case.get("output", "")).rstrip("\n"),
                ]
            )
        )
    return "\n\n".join(parts)


def _is_functional_case(case: dict[str, Any]) -> bool:
    testtype = str(case.get("testtype") or case.get("test_type") or case.get("type") or "stdin").lower()
    return testtype in {"functional", "function", "leetcode"}
