import unittest

from react_bench.benchmarks.evalplus import EvalPlusAdapter
from react_bench.config import RunnerConfig
from react_bench.protocol.schema import Task


class EvalPlusAdapterTests(unittest.TestCase):
    def test_public_feedback_records_parallel_base_and_plus_audit(self):
        task = Task(
            benchmark="evalplus",
            dataset="mini",
            task_id="abs_val",
            language="python",
            prompt="def abs_val(x):\n",
            tests_metadata={
                "prompt": "def abs_val(x):\n",
                "canonical_solution": "    return abs(x)",
                "entry_point": "abs_val",
                "base_input": [1],
                "plus_input": [-1],
            },
        )
        code = "def abs_val(x):\n    return x"
        obs = EvalPlusAdapter().judge(task, code, RunnerConfig(feedback_mode="public"))

        self.assertEqual(obs.status, "pass")
        self.assertIn("status: pass", obs.obs_for_model)
        self.assertEqual(obs.obs_full["evalplus"]["base_status"], "pass")
        self.assertEqual(obs.obs_full["evalplus"]["plus_status"], "fail")


if __name__ == "__main__":
    unittest.main()
