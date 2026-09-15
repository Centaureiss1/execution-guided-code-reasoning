from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParserResult:
    think: str = ""
    submit: str = ""
    status: str = "ok"
    error_type: str | None = None
    warnings: list[str] = field(default_factory=list)


_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
_SUBMIT_RE = re.compile(r"<submit>(.*?)</submit>", flags=re.DOTALL | re.IGNORECASE)
_OBS_RE = re.compile(r"<obs>.*?</obs>", flags=re.DOTALL | re.IGNORECASE)


def parse_model_output(text: str) -> ParserResult:
    result = ParserResult()

    think_match = _THINK_RE.search(text)
    submit_search_start = think_match.end() if think_match else 0
    if not think_match:
        qwen_match = _QWEN_NATIVE_CLOSE_RE.search(text)
        if qwen_match:
            think_match = _QwenNativeThinkMatch(text[: qwen_match.start()], qwen_match.end())
            submit_search_start = qwen_match.end()
    submit_match = _SUBMIT_RE.search(text, pos=submit_search_start)
    if not submit_match:
        result.status = "format_error"
        result.error_type = _missing_submit_error(text, submit_search_start)
    if result.status != "ok":
        return result

    result.think = think_match.group(1).strip() if think_match else ""
    result.submit = _strip_markdown_fence(submit_match.group(1).strip(), result)
    if re.search(r"</?obs>", text[submit_match.end() :], flags=re.IGNORECASE):
        result.status = "format_error"
        result.error_type = "model_generated_obs"
        result.warnings.append("model output contained obs tag after submit")
        return result
    if re.search(r"</?obs>", result.think, flags=re.IGNORECASE):
        result.warnings.append("think_mentioned_obs_tag")
    if not result.submit:
        result.status = "format_error"
        result.error_type = "empty_submit"
    return result


def parse_recovery_output(text: str) -> ParserResult:
    parsed = parse_model_output(text)
    if parsed.status == "ok":
        return parsed
    if parsed.error_type not in {"missing_submit", "incomplete_output"}:
        return parsed

    fallback = _extract_code_fallback(text)
    if not fallback:
        return parsed
    return ParserResult(
        think="",
        submit=fallback,
        status="ok",
        warnings=["recovery_code_without_submit_tag"],
    )


def _missing_submit_error(text: str, pos: int = 0) -> str:
    if re.search(r"<submit>", text[pos:], flags=re.IGNORECASE) and not re.search(
        r"</submit>", text[pos:], flags=re.IGNORECASE
    ):
        return "incomplete_output"
    return "missing_submit"


def _strip_markdown_fence(code: str, result: ParserResult) -> str:
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", code, flags=re.DOTALL | re.IGNORECASE)
    if match:
        result.warnings.append("stripped_markdown_fence")
        return match.group(1).strip()
    return code


def _extract_code_fallback(text: str) -> str:
    fenced = re.fullmatch(r"\s*```(?:python|py)?\s*\n(.*?)\n```\s*", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    stripped = text.strip()
    if not stripped:
        return ""
    if re.search(r"</?think>|</?obs>|</?submit>", stripped, flags=re.IGNORECASE):
        return ""
    if _looks_like_python_code(stripped):
        return stripped
    return ""


def _looks_like_python_code(text: str) -> bool:
    code_markers = [
        r"^\s*import\s+",
        r"^\s*from\s+\S+\s+import\s+",
        r"^\s*def\s+\w+\s*\(",
        r"^\s*class\s+\w+",
        r"^\s*if\s+__name__\s*==",
        r"sys\.stdin",
        r"\bprint\s*\(",
    ]
    return any(re.search(pattern, text, flags=re.MULTILINE) for pattern in code_markers)


_QWEN_NATIVE_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)


class _QwenNativeThinkMatch:
    def __init__(self, think: str, end_pos: int) -> None:
        self._think = think
        self._end_pos = end_pos

    def group(self, index: int) -> str:
        if index != 1:
            raise IndexError(index)
        return self._think

    def end(self) -> int:
        return self._end_pos
