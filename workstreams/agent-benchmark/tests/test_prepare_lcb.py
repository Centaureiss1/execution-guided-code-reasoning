import json
import tempfile
import unittest
from pathlib import Path

from react_bench.benchmarks.livecodebench import LiveCodeBenchAdapter, coerce_lcb_cases
from react_bench.cli import main
from react_bench.prepare import convert_lcb_row
from react_bench.prepare.lcb import _load_subset_ids


class PrepareLcbTests(unittest.TestCase):
    def test_coerce_official_json_string_cases(self):
        cases = coerce_lcb_cases(
            json.dumps(
                [
                    {"input": "1 2\n", "output": "3\n", "testtype": "stdin"},
                    {"input": "4 5\n", "output": "9\n", "testtype": "stdin"},
                ]
            )
        )
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["input"], "1 2\n")
        self.assertEqual(cases[0]["testtype"], "stdin")

    def test_convert_lcb_row_to_runner_schema(self):
        row = {
            "question_id": "sum",
            "question_title": "Sum",
            "question_content": "Read two ints.",
            "platform": "atcoder",
            "public_test_cases": json.dumps(
                [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
            ),
            "private_test_cases": json.dumps(
                [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}]
            ),
            "metadata": "{}",
        }
        converted = convert_lcb_row(row, include_private=True)
        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertEqual(converted["question_id"], "sum")
        self.assertEqual(converted["public_test_cases"][0]["output"], "3\n")
        self.assertEqual(converted["private_test_cases"][0]["output"], "5\n")

    def test_convert_lcb_row_skips_non_stdio_by_default(self):
        row = {
            "question_id": "fn",
            "question_content": "Implement f.",
            "public_test_cases": json.dumps(
                [{"input": "[1, 2]", "output": "3", "testtype": "functional"}]
            ),
        }
        self.assertIsNone(convert_lcb_row(row))

    def test_convert_lcb_row_keeps_functional_when_allowed(self):
        row = {
            "question_id": "fn",
            "question_content": "Implement f.",
            "starter_code": "class Solution:\n    def f(self, nums: List[int]) -> int:\n        pass",
            "public_test_cases": json.dumps(
                [{"input": "[1, 2]", "output": "3", "testtype": "functional"}]
            ),
            "metadata": json.dumps({"func_name": "f"}),
        }
        converted = convert_lcb_row(row, stdin_only=False)
        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertEqual(converted["public_test_cases"][0]["testtype"], "functional")
        self.assertEqual(converted["metadata"]["func_name"], "f")

    def test_converted_jsonl_loads_in_adapter(self):
        row = {
            "question_id": "sum",
            "question_content": "Read two ints.",
            "public_test_cases": json.dumps(
                [{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]
            ),
        }
        converted = convert_lcb_row(row)
        assert converted is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lcb.jsonl"
            path.write_text(json.dumps(converted) + "\n", encoding="utf-8")
            tasks = LiveCodeBenchAdapter(tasks_path=str(path)).load_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertIn("Public examples", tasks[0].prompt)
        self.assertEqual(tasks[0].tests_metadata["public_test_cases"][0]["input"], "1 2\n")

    def test_prepare_lcb_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["prepare-lcb", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_load_subset_ids_from_diverse_subset_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subset.json"
            path.write_text(
                json.dumps(
                    {
                        "question_ids": ["a", "b"],
                        "questions": [{"question_id": "ignored"}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_load_subset_ids(str(path)), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
