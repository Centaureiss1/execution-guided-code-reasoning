import contextlib
import io
import unittest

from react_bench.cli import main


class CliTests(unittest.TestCase):
    def test_run_help_includes_debug_flags(self):
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(stdout):
            main(["run", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--verbose", help_text)
        self.assertIn("--print-submit", help_text)
        self.assertIn("--print-raw", help_text)
        self.assertIn("--no-resume", help_text)


if __name__ == "__main__":
    unittest.main()
