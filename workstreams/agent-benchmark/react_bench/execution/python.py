from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from react_bench.protocol.schema import Observation


@dataclass
class _RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class PythonExecutor:
    def __init__(self, timeout_seconds: float = 10.0, output_limit: int = 4000) -> None:
        self.timeout_seconds = timeout_seconds
        self.output_limit = output_limit

    def run_function_tests(self, code: str, tests_metadata: dict[str, Any]) -> Observation:
        test_units = _extract_function_test_units(tests_metadata)
        if not test_units:
            test_units = _extract_differential_test_units(tests_metadata)
        if not test_units:
            return Observation(
                status="fail",
                error_type="format_error",
                obs_for_model="status: fail error_type: format_error reason: no function tests available",
                obs_full={"status": "fail", "error_type": "format_error", "reason": "no function tests"},
            )

        passed = 0
        first_failure: Observation | None = None
        for test_code in test_units:
            script = code.rstrip() + "\n\n" + test_code.rstrip() + "\n"
            result = self._run_script(script)
            obs = self._observe_process(result)
            if obs.passed:
                passed += 1
            elif first_failure is None:
                first_failure = obs

        if first_failure is None:
            return Observation(
                status="pass",
                error_type=None,
                obs_for_model=f"status: pass passed_tests: {passed} failed_tests: 0",
                obs_full={"status": "pass", "passed_tests": passed, "failed_tests": 0},
            )

        failed = len(test_units) - passed
        full = dict(first_failure.obs_full)
        full.update({"passed_tests": passed, "failed_tests": failed})
        return Observation(
            status="fail",
            error_type=first_failure.error_type,
            obs_for_model=_format_fail_obs(first_failure.error_type, passed, failed, first_failure.obs_full),
            obs_full=full,
        )

    def run_stdio_tests(self, code: str, test_cases: list[dict[str, Any]]) -> Observation:
        if not test_cases:
            return Observation(
                status="fail",
                error_type="format_error",
                obs_for_model="status: fail error_type: format_error reason: no stdio tests available",
                obs_full={"status": "fail", "error_type": "format_error", "reason": "no stdio tests"},
            )

        passed = 0
        first_failure: dict[str, Any] | None = None
        for index, case in enumerate(test_cases):
            result = self._run_program(code, str(case.get("input", "")))
            obs = self._observe_process(result)
            expected = _normalize_output(str(case.get("output", "")))
            actual = _normalize_output(result.stdout)
            if obs.passed and actual == expected:
                passed += 1
                continue
            if first_failure is None:
                if not obs.passed:
                    first_failure = dict(obs.obs_full)
                else:
                    first_failure = {
                        "status": "fail",
                        "error_type": "wrong_answer",
                        "case_index": index,
                        "expected": _truncate(expected, self.output_limit),
                        "actual": _truncate(actual, self.output_limit),
                    }

        if first_failure is None:
            return Observation(
                status="pass",
                error_type=None,
                obs_for_model=f"status: pass passed_tests: {passed} failed_tests: 0",
                obs_full={"status": "pass", "passed_tests": passed, "failed_tests": 0},
            )

        failed = len(test_cases) - passed
        error_type = str(first_failure.get("error_type", "wrong_answer"))
        first_failure.update({"passed_tests": passed, "failed_tests": failed})
        return Observation(
            status="fail",
            error_type=error_type,
            obs_for_model=_format_fail_obs(error_type, passed, failed, first_failure),
            obs_full=first_failure,
        )

    def run_lcb_functional_tests(
        self,
        code: str,
        func_name: str,
        test_cases: list[dict[str, Any]],
    ) -> Observation:
        if not func_name:
            return Observation(
                status="fail",
                error_type="format_error",
                obs_for_model="status: fail error_type: format_error reason: missing functional entry point",
                obs_full={"status": "fail", "error_type": "format_error", "reason": "missing func_name"},
            )
        if not test_cases:
            return Observation(
                status="fail",
                error_type="format_error",
                obs_for_model="status: fail error_type: format_error reason: no functional tests available",
                obs_full={"status": "fail", "error_type": "format_error", "reason": "no functional tests"},
            )

        result = self._run_lcb_functional_runner(code, func_name, test_cases)
        if result.timed_out:
            return Observation(
                status="fail",
                error_type="timeout",
                obs_for_model="status: fail error_type: timeout",
                obs_full={"status": "fail", "error_type": "timeout", "stdout": result.stdout, "stderr": result.stderr},
            )

        summary = _extract_runner_summary(result.stdout)
        if summary is None:
            return self._observe_process(result)

        passed = int(summary.get("passed_tests") or 0)
        failed = int(summary.get("failed_tests") or 0)
        if summary.get("status") == "pass":
            return Observation(
                status="pass",
                error_type=None,
                obs_for_model=f"status: pass passed_tests: {passed} failed_tests: 0",
                obs_full={"status": "pass", "passed_tests": passed, "failed_tests": 0},
            )

        error_type = str(summary.get("error_type") or "wrong_answer")
        details = {
            "status": "fail",
            "error_type": error_type,
            "passed_tests": passed,
            "failed_tests": failed,
            "case_index": summary.get("case_index"),
            "expected": summary.get("expected"),
            "actual": summary.get("actual"),
            "stderr": summary.get("traceback") or result.stderr,
            "stdout": result.stdout,
        }
        return Observation(
            status="fail",
            error_type=error_type,
            obs_for_model=_format_fail_obs(error_type, passed, failed, details),
            obs_full=details,
        )

    def run_module_tests(self, code: str, tests_metadata: dict[str, Any]) -> Observation:
        test_code = str(tests_metadata.get("test") or "")
        if not test_code.strip():
            return Observation(
                status="fail",
                error_type="format_error",
                obs_for_model="status: fail error_type: format_error reason: no module tests available",
                obs_full={"status": "fail", "error_type": "format_error", "reason": "no module tests"},
            )

        result = self._run_solution_module_tests(code, test_code)
        if result.timed_out:
            return Observation(
                status="fail",
                error_type="timeout",
                obs_for_model="status: fail error_type: timeout",
                obs_full={"status": "fail", "error_type": "timeout", "stdout": result.stdout, "stderr": result.stderr},
            )

        summary = _extract_runner_summary(result.stdout)
        if summary is None:
            return self._observe_process(result)

        passed = int(summary.get("passed_tests") or 0)
        failed = int(summary.get("failed_tests") or 0)
        if summary.get("status") == "pass":
            return Observation(
                status="pass",
                error_type=None,
                obs_for_model=f"status: pass passed_tests: {passed} failed_tests: 0",
                obs_full={"status": "pass", "passed_tests": passed, "failed_tests": 0},
            )

        error_type = str(summary.get("error_type") or "runtime_error")
        details = {
            "status": "fail",
            "error_type": error_type,
            "passed_tests": passed,
            "failed_tests": failed,
            "test_name": summary.get("test_name"),
            "stderr": summary.get("traceback") or result.stderr,
            "stdout": result.stdout,
        }
        return Observation(
            status="fail",
            error_type=error_type,
            obs_for_model=_format_fail_obs(error_type, passed, failed, details),
            obs_full=details,
        )

    def _run_script(self, script: str) -> _RunResult:
        return self._run_program(script, stdin="")

    def _run_solution_module_tests(self, code: str, test_code: str) -> _RunResult:
        with tempfile.TemporaryDirectory(prefix="react_bench_") as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "solution.py").write_text(code, encoding="utf-8")
            (tmp_path / "test_solution.py").write_text(test_code, encoding="utf-8")
            (tmp_path / "pytest.py").write_text(_PYTEST_SHIM, encoding="utf-8")
            runner_path = tmp_path / "_react_bench_test_runner.py"
            runner_path.write_text(_MODULE_TEST_RUNNER, encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            try:
                completed = subprocess.run(
                    [sys.executable, str(runner_path)],
                    text=True,
                    capture_output=True,
                    cwd=tmpdir,
                    env=env,
                    timeout=self.timeout_seconds,
                )
                return _RunResult(
                    returncode=completed.returncode,
                    stdout=_truncate(completed.stdout, self.output_limit),
                    stderr=_truncate(completed.stderr, self.output_limit),
                )
            except subprocess.TimeoutExpired as exc:
                return _RunResult(
                    returncode=-1,
                    stdout=_truncate(exc.stdout or "", self.output_limit),
                    stderr=_truncate(exc.stderr or "", self.output_limit),
                    timed_out=True,
                )

    def _run_lcb_functional_runner(
        self,
        code: str,
        func_name: str,
        test_cases: list[dict[str, Any]],
    ) -> _RunResult:
        with tempfile.TemporaryDirectory(prefix="react_bench_") as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "solution.py").write_text(code, encoding="utf-8")
            runner_path = tmp_path / "_react_bench_lcb_functional_runner.py"
            runner_path.write_text(_LCB_FUNCTIONAL_RUNNER, encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            try:
                completed = subprocess.run(
                    [sys.executable, str(runner_path), func_name],
                    input=__import__("json").dumps(test_cases),
                    text=True,
                    capture_output=True,
                    cwd=tmpdir,
                    env=env,
                    timeout=self.timeout_seconds,
                )
                return _RunResult(
                    returncode=completed.returncode,
                    stdout=_truncate(completed.stdout, self.output_limit),
                    stderr=_truncate(completed.stderr, self.output_limit),
                )
            except subprocess.TimeoutExpired as exc:
                return _RunResult(
                    returncode=-1,
                    stdout=_truncate(exc.stdout or "", self.output_limit),
                    stderr=_truncate(exc.stderr or "", self.output_limit),
                    timed_out=True,
                )

    def _run_program(self, code: str, stdin: str) -> _RunResult:
        with tempfile.TemporaryDirectory(prefix="react_bench_") as tmpdir:
            path = Path(tmpdir) / "solution.py"
            path.write_text(code, encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            try:
                completed = subprocess.run(
                    [sys.executable, str(path)],
                    input=stdin,
                    text=True,
                    capture_output=True,
                    cwd=tmpdir,
                    env=env,
                    timeout=self.timeout_seconds,
                )
                return _RunResult(
                    returncode=completed.returncode,
                    stdout=_truncate(completed.stdout, self.output_limit),
                    stderr=_truncate(completed.stderr, self.output_limit),
                )
            except subprocess.TimeoutExpired as exc:
                return _RunResult(
                    returncode=-1,
                    stdout=_truncate(exc.stdout or "", self.output_limit),
                    stderr=_truncate(exc.stderr or "", self.output_limit),
                    timed_out=True,
                )

    def _observe_process(self, result: _RunResult) -> Observation:
        if result.timed_out:
            return Observation(
                status="fail",
                error_type="timeout",
                obs_for_model="status: fail error_type: timeout",
                obs_full={"status": "fail", "error_type": "timeout", "stdout": result.stdout, "stderr": result.stderr},
            )
        if result.returncode == 0:
            return Observation(
                status="pass",
                error_type=None,
                obs_for_model="status: pass",
                obs_full={"status": "pass", "stdout": result.stdout, "stderr": result.stderr},
            )

        error_type = _classify_stderr(result.stderr)
        return Observation(
            status="fail",
            error_type=error_type,
            obs_for_model=f"status: fail error_type: {error_type} stderr_tail: {_one_line(result.stderr)}",
            obs_full={
                "status": "fail",
                "error_type": error_type,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )


def _extract_function_test_units(tests_metadata: dict[str, Any]) -> list[str]:
    for key in ("test_list", "tests"):
        tests = tests_metadata.get(key)
        if isinstance(tests, list):
            return [str(test) for test in tests]

    test = tests_metadata.get("test")
    entry_point = tests_metadata.get("entry_point")
    if isinstance(test, str) and test.strip():
        if entry_point and "def check(" in test and "check(" not in test.split("def check(", 1)[1]:
            return [test + f"\n\ncheck({entry_point})"]
        if entry_point and "def check(" in test:
            return [test + f"\n\ncheck({entry_point})"]
        return [test]
    return []


def _extract_differential_test_units(tests_metadata: dict[str, Any]) -> list[str]:
    entry_point = tests_metadata.get("entry_point")
    canonical_solution = tests_metadata.get("canonical_solution")
    prompt = tests_metadata.get("prompt") or ""
    inputs = tests_metadata.get("generated_inputs") or []
    if not entry_point or not canonical_solution or not inputs:
        return []

    reference_code = str(prompt).rstrip() + "\n" + str(canonical_solution).rstrip() + "\n"
    units = []
    for input_value in inputs:
        units.append(
            "\n".join(
                [
                    "import math",
                    f"_reference_code = {reference_code!r}",
                    "_reference_ns = {}",
                    "exec(_reference_code, _reference_ns)",
                    f"_args = {input_value!r}",
                    "if isinstance(_args, dict):",
                    f"    _expected = _reference_ns[{str(entry_point)!r}](**_args)",
                    f"    _actual = {entry_point}(**_args)",
                    "else:",
                    "    if not isinstance(_args, (tuple, list)):",
                    "        _args = (_args,)",
                    f"    _expected = _reference_ns[{str(entry_point)!r}](*_args)",
                    f"    _actual = {entry_point}(*_args)",
                    "assert _actual == _expected, f'expected {_expected!r}, got {_actual!r}'",
                ]
            )
        )
    return units


def _classify_stderr(stderr: str) -> str:
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return "syntax_error"
    if "AssertionError" in stderr:
        return "wrong_answer"
    return "runtime_error"


def _format_fail_obs(error_type: str | None, passed: int, failed: int, details: dict[str, Any]) -> str:
    chunks = [
        "status: fail",
        f"error_type: {error_type or 'unknown'}",
        f"passed_tests: {passed}",
        f"failed_tests: {failed}",
    ]
    if details.get("case_index") is not None:
        chunks.append(f"case_index: {details['case_index']}")
    if details.get("stderr"):
        chunks.append(f"stderr_tail: {_one_line(str(details['stderr']))}")
    if details.get("expected") is not None:
        chunks.append(f"expected: {_one_line(str(details['expected']))}")
    if details.get("actual") is not None:
        chunks.append(f"actual: {_one_line(str(details['actual']))}")
    return " ".join(chunks)


def _normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _extract_runner_summary(stdout: str) -> dict[str, Any] | None:
    prefix = "__REACT_BENCH_RESULT__"
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            import json

            return json.loads(line[len(prefix) :])
    return None


def _truncate(text: str | bytes, limit: int) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _one_line(text: str, limit: int = 400) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[-limit:]


_PYTEST_SHIM = r'''
from __future__ import annotations

import math
import os
import re


class RaisesContext:
    def __init__(self, expected_exception, match=None):
        self.expected_exception = expected_exception
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected_exception}")
        if not issubclass(exc_type, self.expected_exception):
            return False
        self.value = exc
        if self.match is not None and re.search(self.match, str(exc)) is None:
            raise AssertionError(f"exception message {exc!r} does not match {self.match!r}")
        return True


def raises(expected_exception, match=None):
    return RaisesContext(expected_exception, match=match)


def fail(message=""):
    raise AssertionError(message)


class Approx:
    def __init__(self, expected, rel=None, abs=None, nan_ok=False):
        self.expected = expected
        self.rel = 1e-6 if rel is None else rel
        self.abs = 1e-12 if abs is None else abs
        self.nan_ok = nan_ok

    def __eq__(self, actual):
        return _approx_equal(actual, self.expected, self.rel, self.abs, self.nan_ok)


def approx(expected, rel=None, abs=None, nan_ok=False):
    return Approx(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def _approx_equal(actual, expected, rel, abs_tol, nan_ok):
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(_approx_equal(a, e, rel, abs_tol, nan_ok) for a, e in zip(actual, expected))
        )
    try:
        if nan_ok and math.isnan(actual) and math.isnan(expected):
            return True
    except TypeError:
        pass
    try:
        return math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol)
    except TypeError:
        return actual == expected


def fixture(func=None, **kwargs):
    def decorate(inner):
        inner._react_pytest_fixture = True
        return inner
    if func is None:
        return decorate
    return decorate(func)


class _Mark:
    def parametrize(self, names, values):
        def decorate(func):
            current = getattr(func, "_react_parametrize", [])
            current.append((names, list(values)))
            func._react_parametrize = current
            return func
        return decorate

    def skipif(self, condition, reason=None):
        def decorate(func):
            if condition:
                func._react_skip = reason or "skipif"
            return func
        return decorate

    def skip(self, reason=None):
        def decorate(func):
            func._react_skip = reason or "skip"
            return func
        return decorate


mark = _Mark()


def skip(reason=None):
    raise SystemExit(reason or "skip")


class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name=None, value=None):
        if value is None and isinstance(target, str):
            module_name, attr = target.rsplit(".", 1)
            module = __import__(module_name, fromlist=[attr])
            old = getattr(module, attr)
            self._undo.append(lambda: setattr(module, attr, old))
            setattr(module, attr, name)
            return
        old = getattr(target, name)
        self._undo.append(lambda: setattr(target, name, old))
        setattr(target, name, value)

    def setitem(self, mapping, name, value):
        exists = name in mapping
        old = mapping.get(name)
        self._undo.append(lambda: mapping.__setitem__(name, old) if exists else mapping.__delitem__(name))
        mapping[name] = value

    def setenv(self, name, value):
        old = os.environ.get(name)
        exists = name in os.environ
        self._undo.append(lambda: os.environ.__setitem__(name, old) if exists else os.environ.pop(name, None))
        os.environ[name] = str(value)

    def chdir(self, path):
        old = os.getcwd()
        self._undo.append(lambda: os.chdir(old))
        os.chdir(path)

    def undo(self):
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()
'''


_MODULE_TEST_RUNNER = r'''
from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import itertools
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

RESULT_PREFIX = "__REACT_BENCH_RESULT__"


class CaptureFixture:
    def __init__(self, out_buffer, err_buffer):
        self.out_buffer = out_buffer
        self.err_buffer = err_buffer
        self.out_pos = len(out_buffer.getvalue())
        self.err_pos = len(err_buffer.getvalue())

    def readouterr(self):
        out = self.out_buffer.getvalue()[self.out_pos :]
        err = self.err_buffer.getvalue()[self.err_pos :]
        self.out_pos = len(self.out_buffer.getvalue())
        self.err_pos = len(self.err_buffer.getvalue())
        return SimpleNamespace(out=out, err=err)


class TmpDir:
    def __init__(self, path):
        self.path = Path(path)

    def __str__(self):
        return str(self.path)

    def __fspath__(self):
        return str(self.path)

    @property
    def strpath(self):
        return str(self.path)

    def join(self, *parts):
        return TmpDir(self.path.joinpath(*parts))

    def mkdir(self, *parts):
        path = self.path.joinpath(*parts) if parts else self.path
        path.mkdir(parents=True, exist_ok=True)
        return TmpDir(path)

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")

    def read(self):
        return self.path.read_text(encoding="utf-8")

    def write_text(self, text, encoding="utf-8"):
        self.path.write_text(text, encoding=encoding)

    def read_text(self, encoding="utf-8"):
        return self.path.read_text(encoding=encoding)


def classify(exc):
    if isinstance(exc, (SyntaxError, IndentationError)):
        return "syntax_error"
    if isinstance(exc, AssertionError):
        return "wrong_answer"
    return "runtime_error"


def fixture_value(name, module, cache, out_buffer, err_buffer, tmp_root):
    if name in cache:
        return cache[name]
    if name in {"capsys", "capfd"}:
        cache[name] = CaptureFixture(out_buffer, err_buffer)
        return cache[name]
    if name == "tmp_path":
        path = Path(tempfile.mkdtemp(prefix="tmp_path_", dir=tmp_root))
        cache[name] = path
        return path
    if name == "tmpdir":
        path = tempfile.mkdtemp(prefix="tmpdir_", dir=tmp_root)
        cache[name] = TmpDir(path)
        return cache[name]
    if name == "monkeypatch":
        import pytest

        cache[name] = pytest.MonkeyPatch()
        return cache[name]

    provider = getattr(module, name, None)
    if callable(provider) and (
        not name.startswith("test") or getattr(provider, "_react_pytest_fixture", False)
    ):
        kwargs = {
            param.name: fixture_value(param.name, module, cache, out_buffer, err_buffer, tmp_root)
            for param in inspect.signature(provider).parameters.values()
        }
        value = provider(**kwargs)
        cache[name] = value
        return value

    raise TypeError(f"unsupported test fixture or parameter: {name}")


def parametrize_cases(func):
    decorators = getattr(func, "_react_parametrize", [])
    if not decorators:
        return [({}, "")]
    all_groups = []
    for names, values in decorators:
        if isinstance(names, str):
            name_list = [name.strip() for name in names.split(",")]
        else:
            name_list = list(names)
        group = []
        for value in values:
            if len(name_list) == 1:
                tuple_value = (value,)
            else:
                tuple_value = tuple(value)
            group.append(dict(zip(name_list, tuple_value)))
        all_groups.append(group)

    cases = []
    for combo in itertools.product(*all_groups):
        merged = {}
        case_name_parts = []
        for item in combo:
            merged.update(item)
            case_name_parts.extend(f"{key}={value!r}" for key, value in item.items())
        cases.append((merged, "[" + ",".join(case_name_parts) + "]"))
    return cases


def run_one(module, name, func, param_values, out_buffer, err_buffer, tmp_root):
    fixture_cache = {}
    kwargs = dict(param_values)
    for param in inspect.signature(func).parameters.values():
        if param.name not in kwargs:
            kwargs[param.name] = fixture_value(
                param.name, module, fixture_cache, out_buffer, err_buffer, tmp_root
            )
    try:
        with contextlib.redirect_stdout(out_buffer), contextlib.redirect_stderr(err_buffer):
            func(**kwargs)
        return None
    finally:
        monkeypatch = fixture_cache.get("monkeypatch")
        if monkeypatch is not None:
            monkeypatch.undo()


def main():
    summary = {"status": "pass", "passed_tests": 0, "failed_tests": 0}
    out_buffer = io.StringIO()
    err_buffer = io.StringIO()
    tmp_root = os.getcwd()

    try:
        module = importlib.import_module("test_solution")
    except BaseException as exc:
        summary.update(
            {
                "status": "fail",
                "failed_tests": 1,
                "error_type": classify(exc),
                "test_name": "import",
                "traceback": traceback.format_exc(),
            }
        )
        print(RESULT_PREFIX + json.dumps(summary))
        return 1

    try:
        solution = importlib.import_module("solution")
        for attr_name, value in vars(solution).items():
            if not attr_name.startswith("_") and attr_name not in vars(module):
                setattr(module, attr_name, value)
    except BaseException as exc:
        summary.update(
            {
                "status": "fail",
                "failed_tests": 1,
                "error_type": classify(exc),
                "test_name": "import_solution",
                "traceback": traceback.format_exc(),
            }
        )
        print(RESULT_PREFIX + json.dumps(summary))
        return 1

    tests = [
        (name, obj)
        for name, obj in vars(module).items()
        if name.startswith("test") and inspect.isfunction(obj)
    ]
    if not tests:
        summary.update(
            {
                "status": "fail",
                "failed_tests": 1,
                "error_type": "format_error",
                "test_name": "collection",
                "traceback": "no test functions found",
            }
        )
        print(RESULT_PREFIX + json.dumps(summary))
        return 1

    for name, func in tests:
        if getattr(func, "_react_skip", None):
            continue
        for params, suffix in parametrize_cases(func):
            case_name = name + suffix
            try:
                run_one(module, name, func, params, out_buffer, err_buffer, tmp_root)
                summary["passed_tests"] += 1
            except BaseException as exc:
                summary["status"] = "fail"
                summary["failed_tests"] += 1
                if "error_type" not in summary:
                    summary["error_type"] = classify(exc)
                    summary["test_name"] = case_name
                    summary["traceback"] = traceback.format_exc()

    print(RESULT_PREFIX + json.dumps(summary))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


_LCB_FUNCTIONAL_RUNNER = r'''
from __future__ import annotations

import ast
import collections
import functools
import heapq
import itertools
import json
import math
import re
import sys
import traceback
from typing import *

RESULT_PREFIX = "__REACT_BENCH_RESULT__"


def parse_value(text):
    text = str(text).strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(text)
    except Exception:
        pass
    try:
        return float(text)
    except Exception:
        pass
    return text


def parse_args(text):
    lines = [line for line in str(text).splitlines() if line.strip()]
    return [parse_value(line) for line in lines]


def public_repr(value):
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def equal(actual, expected):
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
        except Exception:
            return False
    if isinstance(actual, list) and isinstance(expected, tuple):
        expected = list(expected)
    if isinstance(actual, tuple) and isinstance(expected, list):
        actual = list(actual)
    return actual == expected


def load_solution_namespace():
    source = open("solution.py", encoding="utf-8").read()
    namespace = {
        "__name__": "solution",
        "List": List,
        "Dict": Dict,
        "Set": Set,
        "Tuple": Tuple,
        "Optional": Optional,
        "defaultdict": collections.defaultdict,
        "Counter": collections.Counter,
        "deque": collections.deque,
        "math": math,
        "re": re,
        "itertools": itertools,
        "functools": functools,
        "heapq": heapq,
    }
    exec(compile(source, "solution.py", "exec"), namespace)
    return namespace


def resolve_callable(namespace, func_name):
    solution_class = namespace.get("Solution")
    if solution_class is not None:
        instance = solution_class()
        method = getattr(instance, func_name, None)
        if callable(method):
            return method
    func = namespace.get(func_name)
    if callable(func):
        return func
    raise AttributeError(f"could not find callable Solution().{func_name} or {func_name}")


def classify(exc):
    if isinstance(exc, (SyntaxError, IndentationError)):
        return "syntax_error"
    if isinstance(exc, AssertionError):
        return "wrong_answer"
    if isinstance(exc, AttributeError):
        return "format_error"
    return "runtime_error"


def main():
    func_name = sys.argv[1]
    cases = json.loads(sys.stdin.read() or "[]")
    summary = {"status": "pass", "passed_tests": 0, "failed_tests": 0}
    try:
        namespace = load_solution_namespace()
        target = resolve_callable(namespace, func_name)
        for index, case in enumerate(cases):
            args = parse_args(case.get("input", ""))
            expected = parse_value(case.get("output", ""))
            actual = target(*args)
            if equal(actual, expected):
                summary["passed_tests"] += 1
                continue
            summary.update(
                {
                    "status": "fail",
                    "error_type": "wrong_answer",
                    "case_index": index,
                    "expected": public_repr(expected),
                    "actual": public_repr(actual),
                }
            )
            summary["failed_tests"] = len(cases) - summary["passed_tests"]
            print(RESULT_PREFIX + json.dumps(summary))
            return 1
    except BaseException as exc:
        summary.update(
            {
                "status": "fail",
                "error_type": classify(exc),
                "failed_tests": len(cases) - summary.get("passed_tests", 0),
                "traceback": traceback.format_exc(),
            }
        )
        print(RESULT_PREFIX + json.dumps(summary))
        return 1

    print(RESULT_PREFIX + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
