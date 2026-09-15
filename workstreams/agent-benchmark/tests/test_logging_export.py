import json
import tempfile
import unittest
from pathlib import Path

from react_bench.config import RunnerConfig
from react_bench.distill import distill_failed_prefixes
from react_bench.export import export_sft, select_repair_prefixes
from react_bench.logging import RunLogger, read_jsonl, write_jsonl
from react_bench.protocol.schema import GenerationResult, Task, Trace, Turn
from react_bench.providers.base import BaseProvider


class FakeTeacherProvider(BaseProvider):
    name = "fake_teacher"

    def __init__(self):
        self.model = "fake-teacher"

    def generate(self, messages, config):
        return GenerationResult(
            text="<think>fix addition</think><submit>def add(a, b):\n    return a + b</submit>",
            finish_reason="stop",
        )


class LoggingExportTests(unittest.TestCase):
    def test_failed_prefix_is_exported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(tmpdir)
            task = Task("evalplus", "mini", "t", "python", "prompt", {})
            trace = Trace(task=task, provider="fake", model="fake", config={})
            trace.final_status = "fail"
            logger.log_run(trace)
            self.assertEqual(len(read_jsonl(Path(tmpdir) / "runs.jsonl")), 1)
            self.assertEqual(len(read_jsonl(Path(tmpdir) / "failed_prefixes.jsonl")), 1)

    def test_completed_task_statuses_uses_last_recorded_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(tmpdir)
            task = Task("evalplus", "mini", "t", "python", "prompt", {})

            first = Trace(task=task, provider="fake", model="fake", config={})
            first.final_status = "fail"
            logger.log_run(first)

            second = Trace(task=task, provider="fake", model="fake", config={})
            second.final_status = "pass"
            logger.log_run(second)

            self.assertEqual(logger.completed_task_statuses(), {"t": "pass"})

    def test_logger_round_trips_complex_evalplus_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(tmpdir)
            task = Task(
                "evalplus",
                "mini",
                "complex_task",
                "python",
                "prompt",
                {"base_input": [(1 + 2j,)]},
            )
            trace = Trace(task=task, provider="fake", model="fake", config={})
            trace.final_status = "pass"
            logger.log_run(trace)

            rows = read_jsonl(Path(tmpdir) / "runs.jsonl")
            self.assertEqual(rows[0]["task"]["tests_metadata"]["base_input"][0][0], 1 + 2j)
            self.assertEqual(logger.completed_task_statuses(), {"complex_task": "pass"})

    def test_sft_export_uses_tagged_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(tmpdir)
            task = Task("evalplus", "mini", "t", "python", "prompt", {})
            trace = Trace(task=task, provider="teacher", model="model", config={"feedback_mode": "public"})
            trace.turns.append(
                Turn(
                    1,
                    "",
                    "think",
                    "print(1)",
                    "ok",
                    "status: pass",
                    {"status": "pass"},
                )
            )
            trace.final_status = "pass"
            logger.log_teacher_trace(trace)
            output = Path(tmpdir) / "sft.jsonl"
            count = export_sft(str(Path(tmpdir) / "teacher_traces.jsonl"), str(output))
            self.assertEqual(count, 1)
            row = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertIn("<input>prompt</input>", row["text"])
            self.assertIn("<think>think</think>", row["text"])
            self.assertIn("<submit>print(1)</submit>", row["text"])
            self.assertIn("<obs>status: pass</obs>", row["text"])

    def test_distill_saves_teacher_trace_only_when_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task = Task(
                "evalplus",
                "mini",
                "add",
                "python",
                "Write add(a, b).",
                {"tests": ["assert add(1, 2) == 3"]},
            )
            prefix = Trace(task=task, provider="student", model="student", config={})
            prefix.turns.append(
                Turn(
                    1,
                    "",
                    "bad",
                    "def add(a, b):\n    return a - b",
                    "ok",
                    "status: fail error_type: wrong_answer",
                    {"status": "fail", "error_type": "wrong_answer"},
                )
            )
            prefix.final_status = "fail"
            failed_path = Path(tmpdir) / "failed_prefixes.jsonl"
            write_jsonl(failed_path, [prefix.to_dict()])

            accepted, rejected = distill_failed_prefixes(
                str(failed_path),
                FakeTeacherProvider(),
                RunnerConfig(max_turns=3),
                tmpdir,
                max_teacher_turns=1,
            )

            self.assertEqual((accepted, rejected), (1, 0))
            traces = read_jsonl(Path(tmpdir) / "teacher_traces.jsonl")
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0]["final_status"], "pass")

    def test_select_repair_prefixes_slices_clean_two_failure_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task = Task(
                "evalplus",
                "mini",
                "add",
                "python",
                "Write add(a, b).",
                {"tests": ["assert add(1, 2) == 3"]},
            )
            trace = Trace(task=task, provider="student", model="student", config={})
            trace.turns.extend(
                [
                    Turn(
                        1,
                        "",
                        "bad",
                        "def add(a, b):\n    return a - b",
                        "ok",
                        "status: fail error_type: wrong_answer",
                        {"status": "fail", "error_type": "wrong_answer"},
                        finish_reason="stop",
                    ),
                    Turn(
                        2,
                        "",
                        "still bad",
                        "def add(a, b):\n    return 0",
                        "ok",
                        "status: fail error_type: wrong_answer",
                        {"status": "fail", "error_type": "wrong_answer"},
                        finish_reason="stop",
                    ),
                ]
            )
            trace.final_status = "fail"
            runs_path = Path(tmpdir) / "runs.jsonl"
            output_path = Path(tmpdir) / "repair_prefixes.jsonl"
            write_jsonl(runs_path, [trace.to_dict()])

            stats = select_repair_prefixes(
                str(runs_path),
                str(output_path),
                RunnerConfig(timeout_seconds=1),
            )

            self.assertEqual(stats["selected"], 1)
            rows = read_jsonl(output_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["turns"]), 1)

    def test_select_repair_prefixes_rejects_length_or_format_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task = Task(
                "evalplus",
                "mini",
                "add",
                "python",
                "Write add(a, b).",
                {"tests": ["assert add(1, 2) == 3"]},
            )
            trace = Trace(task=task, provider="student", model="student", config={})
            trace.turns.extend(
                [
                    Turn(
                        1,
                        "",
                        "",
                        "",
                        "format_error",
                        "status: fail error_type: format_error reason: missing_submit",
                        {"status": "fail", "error_type": "format_error", "reason": "missing_submit"},
                        finish_reason="stop",
                    ),
                    Turn(
                        2,
                        "",
                        "bad",
                        "def add(a, b):\n    return a - b",
                        "ok",
                        "status: fail error_type: wrong_answer",
                        {"status": "fail", "error_type": "wrong_answer"},
                        finish_reason="length",
                    ),
                ]
            )
            trace.final_status = "fail"
            runs_path = Path(tmpdir) / "runs.jsonl"
            output_path = Path(tmpdir) / "repair_prefixes.jsonl"
            write_jsonl(runs_path, [trace.to_dict()])

            stats = select_repair_prefixes(
                str(runs_path),
                str(output_path),
                RunnerConfig(timeout_seconds=1),
                rejudge=False,
            )

            self.assertEqual(stats["selected"], 0)
            self.assertEqual(stats["skipped_unclean"], 1)

    def test_select_repair_prefixes_rejects_nonconsecutive_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task = Task(
                "evalplus",
                "mini",
                "add",
                "python",
                "Write add(a, b).",
                {"tests": ["assert add(1, 2) == 3"]},
            )
            trace = Trace(task=task, provider="student", model="student", config={})
            trace.turns.extend(
                [
                    Turn(
                        1,
                        "",
                        "good",
                        "def add(a, b):\n    return a + b",
                        "ok",
                        "status: pass",
                        {"status": "pass"},
                        finish_reason="stop",
                    ),
                    Turn(
                        2,
                        "",
                        "bad",
                        "def add(a, b):\n    return a - b",
                        "ok",
                        "status: fail error_type: wrong_answer",
                        {"status": "fail", "error_type": "wrong_answer"},
                        finish_reason="stop",
                    ),
                    Turn(
                        3,
                        "",
                        "bad",
                        "def add(a, b):\n    return 0",
                        "ok",
                        "status: fail error_type: wrong_answer",
                        {"status": "fail", "error_type": "wrong_answer"},
                        finish_reason="stop",
                    ),
                ]
            )
            trace.final_status = "fail"
            runs_path = Path(tmpdir) / "runs.jsonl"
            output_path = Path(tmpdir) / "repair_prefixes.jsonl"
            write_jsonl(runs_path, [trace.to_dict()])

            stats = select_repair_prefixes(
                str(runs_path),
                str(output_path),
                RunnerConfig(timeout_seconds=1),
            )

            self.assertEqual(stats["selected"], 0)


if __name__ == "__main__":
    unittest.main()
