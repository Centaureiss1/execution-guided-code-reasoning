from __future__ import annotations

from typing import Any

from react_bench.benchmarks.base import BenchmarkAdapter, load_jsonl_tasks
from react_bench.config import RunnerConfig
from react_bench.execution import PythonExecutor
from react_bench.protocol.schema import Observation, Task


class KodCodeAdapter(BenchmarkAdapter):
    benchmark = "kodcode"

    def __init__(
        self,
        dataset: str = "train",
        tasks_path: str | None = None,
        limit: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.tasks_path = tasks_path
        self.limit = limit

    def load_tasks(self) -> list[Task]:
        if self.tasks_path:
            tasks = [self._task_from_row(row) for row in load_jsonl_tasks(self.tasks_path)]
            return tasks[: self.limit] if self.limit else tasks
        tasks = self._load_from_huggingface()
        return tasks[: self.limit] if self.limit else tasks

    def judge(self, task: Task, code: str, config: RunnerConfig) -> Observation:
        executor = PythonExecutor(config.timeout_seconds, config.stderr_stdout_limit)
        return executor.run_module_tests(code, task.tests_metadata)

    def _load_from_huggingface(self) -> list[Task]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The datasets package is required to load KodCode directly. "
                "Install datasets or pass a local JSONL with --tasks."
            ) from exc
        rows = load_dataset("KodCode/KodCode-Light-RL-10K", split=self.dataset)
        return [self._task_from_row(dict(row)) for row in rows]

    def _task_from_row(self, row: dict[str, Any]) -> Task:
        task_id = str(row.get("task_id") or row.get("question_id") or row.get("id"))
        question = str(row.get("prompt") or row.get("question") or "")
        test_info = row.get("test_info") or []
        metadata = {
            "test": row.get("test"),
            "test_info": test_info,
            "reference_solution": row.get("solution"),
            "r1_solution": row.get("r1_solution"),
            "subset": row.get("subset"),
            "difficulty": row.get("gpt_difficulty") or row.get("difficulty"),
            "gpt_pass_percentage": row.get("gpt_pass_percentage"),
            "metadata": row.get("metadata") or {},
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        return Task(
            benchmark=self.benchmark,
            dataset=self.dataset,
            scenario="module_tests",
            task_id=task_id,
            language="python",
            prompt=_render_prompt(question, test_info),
            tests_metadata=metadata,
        )


def _render_prompt(question: str, test_info: Any) -> str:
    parts = [
        question.strip(),
        "",
        "Write a complete Python module. The evaluator will save your submission as solution.py and run unit tests that import from solution.",
    ]
    declarations = _function_declarations(test_info)
    if declarations:
        parts.append("")
        parts.append("Expected public API:")
        parts.extend(f"- {declaration}" for declaration in declarations)
    return "\n".join(part for part in parts if part is not None).strip()


def _function_declarations(test_info: Any) -> list[str]:
    if not isinstance(test_info, list):
        return []
    declarations = []
    for item in test_info:
        if not isinstance(item, dict):
            continue
        declaration = str(item.get("function_declaration") or "").strip()
        if declaration:
            declarations.append(declaration)
    return declarations
