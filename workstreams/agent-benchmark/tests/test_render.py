import unittest

from react_bench.config import RunnerConfig
from react_bench.benchmarks.livecodebench import LiveCodeBenchAdapter
from react_bench.protocol.render import estimate_tokens, render_messages, render_trace_text
from react_bench.protocol.schema import Task, Trace, Turn


class RenderTests(unittest.TestCase):
    def test_prompt_compaction_replaces_old_submit(self):
        task = Task("evalplus", "mini", "t", "python", "short prompt", {"tests": ["assert True"]})
        old_code = "x = 1\n" * 2000
        latest_code = "print('latest')"
        turns = [
            Turn(1, "", "old think", old_code, "ok", "status: fail error_type: wrong_answer", {"status": "fail"}),
            Turn(2, "", "new think", latest_code, "ok", "status: fail error_type: wrong_answer", {"status": "fail"}),
        ]
        rendered = render_trace_text(task, turns, prompt_budget_tokens=700, compact=True)
        self.assertIn("submit_summary", rendered)
        self.assertIn("print('latest')", rendered)

    def test_livecodebench_long_prompt_uses_larger_budget(self):
        cfg = RunnerConfig(prompt_budget_tokens=3000, lcb_long_prompt_budget_tokens=6000, lcb_long_prompt_threshold_tokens=2400)
        self.assertEqual(cfg.effective_prompt_budget("livecodebench", 2500), 6000)
        self.assertEqual(cfg.effective_prompt_budget("evalplus", 2500), 3000)

    def test_render_messages_has_system_and_user(self):
        task = Task("evalplus", "mini", "t", "python", "prompt", {})
        trace = Trace(task=task, provider="fake", model="fake", config={})
        messages = render_messages(task, trace, RunnerConfig())
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("native thinking", messages[0]["content"])
        self.assertNotIn("<think>brief repair reasoning</think>", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertGreaterEqual(estimate_tokens(messages[1]["content"]), 1)

    def test_empty_think_is_not_rendered_back(self):
        task = Task("evalplus", "mini", "t", "python", "prompt", {})
        turns = [Turn(1, "", "", "print(1)", "ok", "status: pass", {"status": "pass"})]
        rendered = render_trace_text(task, turns, prompt_budget_tokens=3000, compact=False)
        self.assertNotIn("<think>", rendered)
        self.assertIn("<submit>print(1)</submit>", rendered)

    def test_lcb_public_examples_are_in_prompt(self):
        adapter = LiveCodeBenchAdapter()
        task = adapter._task_from_row(
            {
                "question_id": "sum",
                "question_content": "Read two ints.",
                "public_test_cases": [{"input": "1 2\n", "output": "3\n"}],
            }
        )
        self.assertIn("Public examples", task.prompt)
        self.assertIn("1 2", task.prompt)
        self.assertIn("3", task.prompt)


if __name__ == "__main__":
    unittest.main()
