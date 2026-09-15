"""Gated code execution reward for Qwen3-8B GRPO training.

Reward:
    0.0 if generated code does not pass all tests.
    2.0 if generated code passes all tests but the final answer format is off.
    2.5 if generated code passes all tests and uses the standard format:

        <think>
        optional reasoning, may be empty
        </think>

        ```python
        complete executable Python code
        ```
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

from swift.rewards.orm import ORM


PYTHON_BLOCK_RE = re.compile(
    r"```python\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
GENERIC_BLOCK_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)
CODE_START_RE = re.compile(
    r"(?m)^((?:from\s+\S+\s+import\s+|import\s+\S+|class\s+\w+|def\s+\w+).*)",
    re.DOTALL,
)
STANDARD_FORMAT_RE = re.compile(
    r"^\s*<think>\s*(.*?)\s*</think>\s*```python\s*(.*?)```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def normalize_tests(tests: Any) -> list[str]:
    if tests is None:
        return []
    if isinstance(tests, str):
        return [tests] if tests.strip() else []
    if isinstance(tests, (list, tuple)):
        return [str(test) for test in tests if str(test).strip()]
    return []


def align_tests_batch(raw_tests_batch: Any, n_completions: int) -> list[Any]:
    if raw_tests_batch is None:
        return [[] for _ in range(n_completions)]
    if isinstance(raw_tests_batch, (list, tuple)):
        if raw_tests_batch and all(isinstance(item, str) for item in raw_tests_batch):
            return [list(raw_tests_batch) for _ in range(n_completions)]
        if len(raw_tests_batch) == n_completions:
            return list(raw_tests_batch)
    return [raw_tests_batch for _ in range(n_completions)]


def extract_code(text: str) -> str | None:
    """Extract executable Python code, using a lenient path for exec reward."""
    tail = text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()

    match = PYTHON_BLOCK_RE.search(tail)
    if match:
        code = match.group(1).strip()
        return code or None

    match = GENERIC_BLOCK_RE.search(tail)
    if match:
        code = match.group(1).strip()
        if re.search(r"\b(def|class)\s+\w+|\bimport\s+\S+|\bfrom\s+\S+\s+import\b", code):
            return code

    match = CODE_START_RE.search(tail)
    if match:
        code = textwrap.dedent(match.group(1)).strip()
        return code or None
    return None


def has_standard_format(text: str) -> bool:
    """Require <think>...</think> followed only by whitespace and a python block."""
    match = STANDARD_FORMAT_RE.fullmatch(text.strip())
    if not match:
        return False
    code = match.group(2).strip()
    return bool(code)


def run_tests(code: str, tests: list[str], timeout: int = 5) -> bool:
    if not tests:
        return False

    program = code + "\n\n" + "\n".join(tests)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.Popen(
                [sys.executable, "-c", program],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=tmpdir,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                return False
        return proc.returncode == 0
    except Exception:
        return False


def reward_one(completion: str, raw_tests: Any) -> float:
    tests = normalize_tests(raw_tests)
    code = extract_code(completion)
    if not code or not run_tests(code, tests):
        return 0.0

    reward = 2.0
    if has_standard_format(completion):
        reward += 0.5
    return reward


def get_reward_workers(n_completions: int) -> int:
    raw_value = os.environ.get("CODE_REWARD_MAX_WORKERS")
    if raw_value:
        try:
            return max(1, min(n_completions, int(raw_value)))
        except ValueError:
            pass
    return max(1, min(n_completions, os.cpu_count() or 1, 8))


class CodeExecFormatGatedReward(ORM):
    """Return 2.0 for passing code, plus 0.5 for the standard final format."""

    def __call__(self, completions: list[str], **kwargs: Any) -> list[float]:
        tests_batch = align_tests_batch(kwargs.get("tests"), len(completions))
        max_workers = get_reward_workers(len(completions))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(reward_one, completions, tests_batch))
