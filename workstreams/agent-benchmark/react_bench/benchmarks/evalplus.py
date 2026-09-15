from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from react_bench.benchmarks.base import BenchmarkAdapter, load_jsonl_tasks
from react_bench.config import RunnerConfig
from react_bench.execution import PythonExecutor
from react_bench.protocol.schema import Observation, Task


class EvalPlusAdapter(BenchmarkAdapter):
    benchmark = "evalplus"

    def __init__(
        self,
        dataset: str = "humaneval",
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
        tasks = self._load_from_evalplus_package()
        return tasks[: self.limit] if self.limit else tasks

    def judge(self, task: Task, code: str, config: RunnerConfig) -> Observation:
        metadata = dict(task.tests_metadata)
        if metadata.get("canonical_solution") and metadata.get("entry_point"):
            base_inputs = list(metadata.get("base_input") or [])
            plus_inputs = list(metadata.get("plus_input") or [])
            if base_inputs or plus_inputs:
                return self._judge_differential_evalplus(
                    code,
                    metadata,
                    base_inputs,
                    plus_inputs,
                    config,
                )

        executor = PythonExecutor(config.timeout_seconds, config.stderr_stdout_limit)
        return executor.run_function_tests(code, metadata)

    def _judge_differential_evalplus(
        self,
        code: str,
        metadata: dict[str, Any],
        base_inputs: list[Any],
        plus_inputs: list[Any],
        config: RunnerConfig,
    ) -> Observation:
        executor = PythonExecutor(config.timeout_seconds, config.stderr_stdout_limit)
        base_metadata = dict(metadata)
        base_metadata["generated_inputs"] = base_inputs or plus_inputs
        plus_metadata = dict(metadata)
        plus_metadata["generated_inputs"] = base_inputs + plus_inputs

        with ThreadPoolExecutor(max_workers=2) as pool:
            base_future = pool.submit(executor.run_function_tests, code, base_metadata)
            plus_future = pool.submit(executor.run_function_tests, code, plus_metadata)
            base_observation = base_future.result()
            plus_observation = plus_future.result()

        feedback_observation = (
            plus_observation
            if config.feedback_mode in {"oracle", "hidden"}
            else base_observation
        )
        obs_full = dict(feedback_observation.obs_full)
        obs_full["evalplus"] = {
            "base_status": base_observation.status,
            "base_error_type": base_observation.error_type,
            "base_passed_tests": base_observation.obs_full.get("passed_tests"),
            "base_failed_tests": base_observation.obs_full.get("failed_tests"),
            "plus_status": plus_observation.status,
            "plus_error_type": plus_observation.error_type,
            "plus_passed_tests": plus_observation.obs_full.get("passed_tests"),
            "plus_failed_tests": plus_observation.obs_full.get("failed_tests"),
        }
        return Observation(
            status=feedback_observation.status,
            error_type=feedback_observation.error_type,
            obs_for_model=feedback_observation.obs_for_model,
            obs_full=obs_full,
        )

    def _load_from_evalplus_package(self) -> list[Task]:
        try:
            from evalplus.data import get_human_eval_plus, get_mbpp_plus
        except ImportError as exc:
            raise RuntimeError(
                "EvalPlus is not installed and no --tasks JSONL was provided. "
                "Install evalplus or pass local task JSONL with prompt/tests fields."
            ) from exc

        dataset = self.dataset.lower()
        if dataset in {"humaneval", "humaneval+"}:
            problems = get_human_eval_plus()
        elif dataset in {"mbpp", "mbpp+"}:
            problems = get_mbpp_plus()
        else:
            raise ValueError(f"Unsupported EvalPlus dataset: {self.dataset}")

        rows = []
        for task_id, problem in problems.items():
            row = dict(problem)
            row.setdefault("task_id", task_id)
            rows.append(self._task_from_row(row))
        return rows

    def _task_from_row(self, row: dict[str, Any]) -> Task:
        task_id = str(row.get("task_id") or row.get("id") or row.get("name"))
        prompt = str(row.get("prompt") or row.get("question") or row.get("instruction") or "")
        tests_metadata = {
            "entry_point": row.get("entry_point"),
            "test": row.get("test"),
            "tests": row.get("tests"),
            "test_list": row.get("test_list"),
            "base_input": row.get("base_input"),
            "plus_input": row.get("plus_input"),
            "canonical_solution": row.get("canonical_solution"),
            "prompt": prompt,
        }
        tests_metadata = {key: value for key, value in tests_metadata.items() if value is not None}
        return Task(
            benchmark=self.benchmark,
            dataset=self.dataset,
            task_id=task_id,
            language="python",
            prompt=prompt,
            tests_metadata=tests_metadata,
        )
