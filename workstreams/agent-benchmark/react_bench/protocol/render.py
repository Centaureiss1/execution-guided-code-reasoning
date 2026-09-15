from __future__ import annotations

from react_bench.config import RunnerConfig
from react_bench.protocol.schema import Task, Trace, Turn


SYSTEM_PROMPT = """You are a benchmark code repair agent.
Use your native thinking format if available.
After thinking, output exactly one final code submission in this format:
<submit>complete Python code only</submit>
Do not output <obs>. Observations are produced by the evaluator.
Do not wrap code in Markdown fences."""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def render_messages(task: Task, trace: Trace, config: RunnerConfig) -> list[dict[str, str]]:
    prompt_budget = config.effective_prompt_budget(task.benchmark, estimate_tokens(task.prompt))
    content = render_trace_text(task, trace.turns, prompt_budget, config.trace_compaction)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def render_trace_text(
    task: Task,
    turns: list[Turn],
    prompt_budget_tokens: int,
    compact: bool = True,
) -> str:
    full = _render_full(task, turns)
    if not compact or estimate_tokens(full) <= prompt_budget_tokens:
        return full
    compacted = _render_compact(task, turns)
    if estimate_tokens(compacted) <= prompt_budget_tokens:
        return compacted
    return _truncate_input(compacted, prompt_budget_tokens)


def _render_full(task: Task, turns: list[Turn]) -> str:
    parts = [f"<input>{task.prompt}</input>"]
    for turn in turns:
        if turn.parsed_think:
            parts.append(f"<think>{turn.parsed_think}</think>")
        parts.append(f"<submit>{turn.parsed_submit}</submit>")
        parts.append(f"<obs>{turn.obs_for_model}</obs>")
    return "\n".join(parts)


def _render_compact(task: Task, turns: list[Turn]) -> str:
    parts = [f"<input>{task.prompt}</input>"]
    if len(turns) > 1:
        for turn in turns[:-1]:
            submit_summary = _summarize_submit(turn.parsed_submit)
            parts.append(
                f'<submit_summary turn="{turn.turn_index}">{submit_summary}</submit_summary>'
            )
            parts.append(f'<obs_summary turn="{turn.turn_index}">{turn.obs_for_model}</obs_summary>')
    if turns:
        latest = turns[-1]
        if latest.parsed_think:
            parts.append(f"<think>{latest.parsed_think}</think>")
        parts.append(f"<submit>{latest.parsed_submit}</submit>")
        parts.append(f"<obs>{latest.obs_for_model}</obs>")
    return "\n".join(parts)


def _summarize_submit(code: str, limit: int = 800) -> str:
    collapsed = "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "\n... [truncated previous submit]"


def _truncate_input(text: str, prompt_budget_tokens: int) -> str:
    max_chars = max(1000, prompt_budget_tokens * 4)
    if len(text) <= max_chars:
        return text
    keep_tail = min(3000, max_chars // 3)
    head = text[: max_chars - keep_tail]
    tail = text[-keep_tail:]
    return head + "\n... [prompt truncated to fit budget]\n" + tail
