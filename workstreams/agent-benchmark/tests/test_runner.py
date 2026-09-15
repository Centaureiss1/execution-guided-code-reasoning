import unittest

from react_bench.agent import ReActRunner
from react_bench.benchmarks.base import BenchmarkAdapter
from react_bench.config import RunnerConfig
from react_bench.execution import PythonExecutor
from react_bench.protocol.schema import GenerationResult, Observation, Task
from react_bench.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    name = "fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.model = "fake-model"

    def generate(self, messages, config, is_recovery=False):
        text = self.outputs.pop(0)
        return GenerationResult(text=text, finish_reason="stop", prompt_tokens=10, completion_tokens=5, latency=0.01)


class SequenceProvider(BaseProvider):
    name = "fake"

    def __init__(self, generations):
        self.generations = list(generations)
        self.model = "fake-model"
        self.seen_max_output_tokens = []
        self.seen_messages = []
        self.seen_is_recovery = []

    def generate(self, messages, config, is_recovery=False):
        self.seen_messages.append(messages)
        self.seen_max_output_tokens.append(config.max_output_tokens)
        self.seen_is_recovery.append(is_recovery)
        return self.generations.pop(0)


class FunctionAdapter(BenchmarkAdapter):
    benchmark = "evalplus"

    def load_tasks(self):
        return []

    def judge(self, task: Task, code: str, config: RunnerConfig) -> Observation:
        return PythonExecutor(config.timeout_seconds).run_function_tests(code, task.tests_metadata)


class CollectingReporter:
    def __init__(self):
        self.events = []

    def turn_start(self, task, turn_index):
        self.events.append(("start", task.task_id, turn_index))

    def turn_end(self, task, turn):
        self.events.append(("end", task.task_id, turn.turn_index, turn.parse_status, turn.obs_full["status"]))


def make_task():
    return Task(
        benchmark="evalplus",
        dataset="mini",
        task_id="add",
        language="python",
        prompt="Write add(a, b).",
        tests_metadata={"tests": ["assert add(1, 2) == 3"]},
    )


class RunnerTests(unittest.TestCase):
    def test_fail_then_pass(self):
        provider = FakeProvider(
            [
                "<think>bad</think><submit>def add(a,b):\n    return a-b</submit>",
                "<think>fix</think><submit>def add(a,b):\n    return a+b</submit>",
            ]
        )
        runner = ReActRunner(FunctionAdapter(), provider, RunnerConfig(max_turns=3))
        trace = runner.run_task(make_task())
        self.assertEqual(trace.final_status, "pass")
        self.assertEqual(len(trace.turns), 2)

    def test_max_turns_one(self):
        provider = FakeProvider(["<think>bad</think><submit>def add(a,b):\n    return a-b</submit>"])
        runner = ReActRunner(FunctionAdapter(), provider, RunnerConfig(max_turns=1))
        trace = runner.run_task(make_task())
        self.assertEqual(trace.final_status, "fail")
        self.assertEqual(len(trace.turns), 1)

    def test_repeated_format_error(self):
        provider = FakeProvider(["no submit here", "still no submit"])
        runner = ReActRunner(FunctionAdapter(), provider, RunnerConfig(max_turns=2))
        trace = runner.run_task(make_task())
        self.assertEqual(trace.final_status, "fail")
        self.assertEqual(trace.turns[0].parse_status, "format_error")
        self.assertEqual(trace.turns[1].parse_status, "format_error")

    def test_reporter_events(self):
        reporter = CollectingReporter()
        provider = FakeProvider(["<submit>def add(a,b):\n    return a+b</submit>"])
        runner = ReActRunner(FunctionAdapter(), provider, RunnerConfig(max_turns=1), reporter=reporter)
        trace = runner.run_task(make_task())
        self.assertEqual(trace.final_status, "pass")
        self.assertEqual(
            reporter.events,
            [
                ("start", "add", 1),
                ("end", "add", 1, "ok", "pass"),
            ],
        )

    def test_length_missing_submit_uses_recovery_same_turn(self):
        provider = SequenceProvider(
            [
                GenerationResult(
                    text="long native reasoning without a submit",
                    finish_reason="length",
                    prompt_tokens=100,
                    completion_tokens=7000,
                    latency=1.0,
                ),
                GenerationResult(
                    text="<submit>def add(a,b):\n    return a+b</submit>",
                    finish_reason="stop",
                    prompt_tokens=200,
                    completion_tokens=20,
                    latency=0.5,
                ),
            ]
        )
        config = RunnerConfig(max_turns=3, max_output_tokens=7000, recovery_max_output_tokens=1234)
        runner = ReActRunner(FunctionAdapter(), provider, config)
        trace = runner.run_task(make_task())
        self.assertEqual(trace.final_status, "pass")
        self.assertEqual(len(trace.turns), 1)
        turn = trace.turns[0]
        self.assertTrue(turn.recovery_used)
        self.assertEqual(turn.recovery_parse_status, "ok")
        self.assertEqual(turn.parsed_submit, "def add(a,b):\n    return a+b")
        self.assertEqual(provider.seen_max_output_tokens, [7000, 1234])
        self.assertEqual(provider.seen_is_recovery, [False, True])
        self.assertTrue(turn.recovery_disable_thinking)
        self.assertIn("DO NOT THINK", provider.seen_messages[1][0]["content"])
        self.assertIn("<reasoning_tail>", provider.seen_messages[1][1]["content"])
        self.assertNotIn("<truncated_reasoning>", provider.seen_messages[1][1]["content"])

    def test_recovery_failure_records_specific_format_error(self):
        provider = SequenceProvider(
            [
                GenerationResult(text="long native reasoning", finish_reason="length"),
                GenerationResult(text="still no submit", finish_reason="length"),
            ]
        )
        runner = ReActRunner(FunctionAdapter(), provider, RunnerConfig(max_turns=3))
        trace = runner.run_task(make_task())
        self.assertEqual(trace.final_status, "fail")
        self.assertEqual(len(trace.turns), 1)
        turn = trace.turns[0]
        self.assertTrue(turn.recovery_used)
        self.assertEqual(turn.obs_full["reason"], "missing_submit_after_recovery")

    def test_recovery_fenced_code_is_executed(self):
        provider = SequenceProvider(
            [
                GenerationResult(text="long native reasoning", finish_reason="length"),
                GenerationResult(text="```python\ndef add(a,b):\n    return a+b\n```", finish_reason="stop"),
            ]
        )
        runner = ReActRunner(FunctionAdapter(), provider, RunnerConfig(max_turns=3))
        trace = runner.run_task(make_task())
        self.assertEqual(trace.final_status, "pass")
        self.assertEqual(len(trace.turns), 1)
        self.assertEqual(trace.turns[0].recovery_parse_status, "ok")
        self.assertEqual(trace.turns[0].parsed_submit, "def add(a,b):\n    return a+b")


if __name__ == "__main__":
    unittest.main()
