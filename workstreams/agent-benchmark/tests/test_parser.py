import unittest

from react_bench.protocol import parse_model_output, parse_recovery_output


class ParserTests(unittest.TestCase):
    def test_valid_output(self):
        parsed = parse_model_output("<think>fix it</think><submit>print(1)</submit>")
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.think, "fix it")
        self.assertEqual(parsed.submit, "print(1)")

    def test_submit_only_is_valid_with_empty_think(self):
        parsed = parse_model_output("<submit>print(1)</submit>")
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.think, "")
        self.assertEqual(parsed.submit, "print(1)")

    def test_missing_submit(self):
        parsed = parse_model_output("<think>fix it</think>")
        self.assertEqual(parsed.status, "format_error")
        self.assertEqual(parsed.error_type, "missing_submit")

    def test_unclosed_submit(self):
        parsed = parse_model_output("<think>fix it</think><submit>print(1)")
        self.assertEqual(parsed.status, "format_error")
        self.assertEqual(parsed.error_type, "incomplete_output")

    def test_model_generated_obs_is_format_error(self):
        parsed = parse_model_output("<think>x</think><submit>print(1)</submit><obs>status: pass</obs>")
        self.assertEqual(parsed.status, "format_error")
        self.assertEqual(parsed.error_type, "model_generated_obs")

    def test_obs_mention_in_thinking_is_allowed(self):
        parsed = parse_model_output("Do not output <obs> in final.\n</think>\n<submit>print(1)</submit>")
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.submit, "print(1)")
        self.assertIn("think_mentioned_obs_tag", parsed.warnings)

    def test_recovery_accepts_fenced_python(self):
        parsed = parse_recovery_output("```python\nprint(1)\n```")
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.submit, "print(1)")
        self.assertIn("recovery_code_without_submit_tag", parsed.warnings)

    def test_recovery_accepts_plain_python_code(self):
        parsed = parse_recovery_output("import sys\nprint(sys.stdin.read())")
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.submit, "import sys\nprint(sys.stdin.read())")

    def test_normal_parser_still_rejects_fenced_without_submit(self):
        parsed = parse_model_output("```python\nprint(1)\n```")
        self.assertEqual(parsed.status, "format_error")
        self.assertEqual(parsed.error_type, "missing_submit")

    def test_markdown_fence_is_unwrapped(self):
        parsed = parse_model_output("<think>x</think><submit>```python\nprint(1)\n```</submit>")
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.submit, "print(1)")
        self.assertIn("stripped_markdown_fence", parsed.warnings)

    def test_qwen_native_thinking_missing_open_tag(self):
        parsed = parse_model_output(
            "reasoning mentions <submit>complete Python code only</submit>\n</think>\n<submit>print(1)</submit>"
        )
        self.assertEqual(parsed.status, "ok")
        self.assertIn("reasoning mentions", parsed.think)
        self.assertEqual(parsed.submit, "print(1)")


if __name__ == "__main__":
    unittest.main()
