import contextlib
import io
import unittest

from react_bench.benchmarks.kodcode import KodCodeAdapter
from react_bench.cli import main
from react_bench.prepare.kodcode import convert_kodcode_row


class PrepareKodCodeTests(unittest.TestCase):
    def test_convert_kodcode_row(self):
        row = {
            "question_id": "Filter_1_I",
            "question": "Return x + 1.",
            "solution": "def inc(x):\n    return x + 1",
            "test": "from solution import inc\n\ndef test_inc():\n    assert inc(1) == 2",
            "test_info": [{"function_declaration": "def inc(x):"}],
            "subset": "Filter",
            "gpt_difficulty": "easy",
            "gpt_pass_percentage": 1.0,
        }
        converted = convert_kodcode_row(row)
        self.assertIsNotNone(converted)
        self.assertEqual(converted["question_id"], "Filter_1_I")
        self.assertEqual(converted["test_info"][0]["function_declaration"], "def inc(x):")

    def test_convert_kodcode_filters(self):
        row = {
            "question_id": "Filter_1_I",
            "question": "Return x + 1.",
            "test": "def test_x(): pass",
            "subset": "Filter",
            "gpt_difficulty": "easy",
            "gpt_pass_percentage": 0.2,
        }
        self.assertIsNone(convert_kodcode_row(row, subset="Algorithm"))
        self.assertIsNone(convert_kodcode_row(row, difficulty="hard"))
        self.assertIsNone(convert_kodcode_row(row, min_gpt_pass_percentage=0.5))

    def test_adapter_builds_module_test_task(self):
        adapter = KodCodeAdapter(tasks_path="unused")
        task = adapter._task_from_row(
            {
                "question_id": "Filter_1_I",
                "question": "Return x + 1.",
                "test": "from solution import inc\n\ndef test_inc():\n    assert inc(1) == 2",
                "test_info": [{"function_declaration": "def inc(x):"}],
                "gpt_difficulty": "easy",
            }
        )
        self.assertEqual(task.benchmark, "kodcode")
        self.assertEqual(task.scenario, "module_tests")
        self.assertIn("def inc(x):", task.prompt)
        self.assertIn("solution.py", task.prompt)

    def test_prepare_kodcode_help(self):
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(stdout):
            main(["prepare-kodcode", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--difficulty", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
