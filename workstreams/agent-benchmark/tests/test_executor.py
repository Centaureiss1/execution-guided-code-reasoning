import unittest

from react_bench.execution import PythonExecutor


class PythonExecutorTests(unittest.TestCase):
    def test_function_pass(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_function_tests(
            "def add(a, b):\n    return a + b",
            {"tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]},
        )
        self.assertEqual(obs.status, "pass")
        self.assertEqual(obs.obs_full["passed_tests"], 2)

    def test_wrong_answer(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_function_tests(
            "def add(a, b):\n    return a - b",
            {"tests": ["assert add(1, 2) == 3"]},
        )
        self.assertEqual(obs.status, "fail")
        self.assertEqual(obs.error_type, "wrong_answer")

    def test_syntax_error(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_function_tests("def broken(:\n    pass", {"tests": ["assert True"]})
        self.assertEqual(obs.status, "fail")
        self.assertEqual(obs.error_type, "syntax_error")

    def test_runtime_error(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_function_tests(
            "def boom():\n    raise ValueError('x')",
            {"tests": ["boom()"]},
        )
        self.assertEqual(obs.status, "fail")
        self.assertEqual(obs.error_type, "runtime_error")

    def test_timeout(self):
        executor = PythonExecutor(timeout_seconds=0.2)
        obs = executor.run_function_tests("while True:\n    pass", {"tests": ["assert True"]})
        self.assertEqual(obs.status, "fail")
        self.assertEqual(obs.error_type, "timeout")

    def test_stdio_wrong_answer(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_stdio_tests("print('no')", [{"input": "", "output": "yes\n"}])
        self.assertEqual(obs.status, "fail")
        self.assertEqual(obs.error_type, "wrong_answer")

    def test_lcb_functional_solution_class_pass(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_lcb_functional_tests(
            "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a + b",
            "add",
            [
                {"input": "1\n2", "output": "3", "testtype": "functional"},
                {"input": "-1\n1", "output": "0", "testtype": "functional"},
            ],
        )
        self.assertEqual(obs.status, "pass")
        self.assertEqual(obs.obs_full["passed_tests"], 2)

    def test_lcb_functional_reports_expected_actual(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_lcb_functional_tests(
            "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a - b",
            "add",
            [{"input": "1\n2", "output": "3", "testtype": "functional"}],
        )
        self.assertEqual(obs.status, "fail")
        self.assertEqual(obs.error_type, "wrong_answer")
        self.assertIn("expected", obs.obs_for_model)
        self.assertIn("actual", obs.obs_for_model)

    def test_differential_function_tests(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_function_tests(
            "def add(a, b):\n    return a + b",
            {
                "prompt": "def add(a, b):\n",
                "canonical_solution": "    return a + b",
                "entry_point": "add",
                "generated_inputs": [(1, 2), (-1, 1)],
            },
        )
        self.assertEqual(obs.status, "pass")
        self.assertEqual(obs.obs_full["passed_tests"], 2)

    def test_module_tests_import_solution_and_call_tests(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_module_tests(
            "def add(a, b):\n    return a + b",
            {"test": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
        )
        self.assertEqual(obs.status, "pass")
        self.assertEqual(obs.obs_full["passed_tests"], 1)

    def test_module_tests_wrong_answer(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_module_tests(
            "def add(a, b):\n    return a - b",
            {"test": "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"},
        )
        self.assertEqual(obs.status, "fail")
        self.assertEqual(obs.error_type, "wrong_answer")

    def test_module_tests_pytest_raises_shim(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_module_tests(
            "def check(value):\n    if value < 0:\n        raise ValueError('negative')",
            {
                "test": "\n".join(
                    [
                        "import pytest",
                        "from solution import check",
                        "",
                        "def test_negative():",
                        "    with pytest.raises(ValueError, match='negative'):",
                        "        check(-1)",
                    ]
                )
            },
        )
        self.assertEqual(obs.status, "pass")

    def test_module_tests_parametrize_shim(self):
        executor = PythonExecutor(timeout_seconds=1)
        obs = executor.run_module_tests(
            "def add(a, b):\n    return a + b",
            {
                "test": "\n".join(
                    [
                        "import pytest",
                        "from solution import add",
                        "",
                        "@pytest.mark.parametrize('a,b,expected', [(1, 2, 3), (-1, 1, 0)])",
                        "def test_add(a, b, expected):",
                        "    assert add(a, b) == expected",
                    ]
                )
            },
        )
        self.assertEqual(obs.status, "pass")
        self.assertEqual(obs.obs_full["passed_tests"], 2)


if __name__ == "__main__":
    unittest.main()
