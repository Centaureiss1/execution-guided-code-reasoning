from __future__ import annotations

from dataclasses import asdict, replace
from typing import Protocol

from react_bench.benchmarks.base import BenchmarkAdapter
from react_bench.config import RunnerConfig
from react_bench.protocol.parser import parse_model_output, parse_recovery_output
from react_bench.protocol.render import render_messages
from react_bench.protocol.schema import GenerationResult, Observation, Task, Trace, Turn
from react_bench.providers.base import BaseProvider


class RunnerReporter(Protocol):
    def turn_start(self, task: Task, turn_index: int) -> None:
        ...

    def turn_end(self, task: Task, turn: Turn) -> None:
        ...


class ReActRunner:
    def __init__(
        self,
        adapter: BenchmarkAdapter,
        provider: BaseProvider,
        config: RunnerConfig,
        reporter: RunnerReporter | None = None,
    ) -> None:
        self.adapter = adapter
        self.provider = provider
        self.config = config
        self.reporter = reporter

    def run_task(self, task: Task, prefix: Trace | None = None) -> Trace:
        trace = prefix or Trace(
            task=task,
            provider=self.provider.name,
            model=self.provider.model,
            config=asdict(self.config),
        )
        start_turn = len(trace.turns) + 1
        for turn_index in range(start_turn, self.config.max_turns + 1):
            messages = render_messages(task, trace, self.config)
            if self.reporter:
                self.reporter.turn_start(task, turn_index)
            generation = self.provider.generate(messages, self.config)
            parsed = parse_model_output(generation.text)
            recovery = None
            recovery_parsed = None
            recovery_config = None
            if _should_recover_submit(generation, parsed.error_type, self.config):
                recovery_config = replace(self.config, max_output_tokens=self.config.recovery_max_output_tokens)
                recovery_messages = _render_recovery_messages(task, generation.text, self.config)
                recovery = self.provider.generate(recovery_messages, recovery_config, is_recovery=True)
                recovery_parsed = parse_recovery_output(recovery.text)
                if recovery_parsed.status == "ok":
                    parsed = recovery_parsed
            if parsed.status != "ok":
                error_type = parsed.error_type or "format_error"
                if recovery is not None and error_type in {"missing_submit", "incomplete_output"}:
                    error_type = "missing_submit_after_recovery"
                observation = _format_error_observation(error_type)
                submit = parsed.submit
                think = parsed.think
            else:
                think = parsed.think
                submit = parsed.submit
                observation = self.adapter.judge(task, submit, self.config)
            if recovery is not None and recovery_parsed is not None:
                observation.obs_full["recovery_used"] = True
                observation.obs_full["recovery_parse_status"] = recovery_parsed.status
                observation.obs_full["recovery_finish_reason"] = recovery.finish_reason
                observation.obs_full["recovery_disable_thinking"] = self.config.recovery_disable_thinking

            turn = Turn(
                turn_index=turn_index,
                raw_output=generation.text,
                parsed_think=think,
                parsed_submit=submit,
                parse_status=parsed.status,
                obs_for_model=observation.obs_for_model,
                obs_full=observation.obs_full,
                token_usage={
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                },
                latency=generation.latency,
                finish_reason=generation.finish_reason,
                recovery_used=recovery is not None,
                recovery_raw_output=recovery.text if recovery else "",
                recovery_parse_status=recovery_parsed.status if recovery_parsed else None,
                recovery_finish_reason=recovery.finish_reason if recovery else None,
                recovery_token_usage={
                    "prompt_tokens": recovery.prompt_tokens if recovery else None,
                    "completion_tokens": recovery.completion_tokens if recovery else None,
                }
                if recovery
                else {},
                recovery_disable_thinking=bool(recovery and self.config.recovery_disable_thinking),
            )
            trace.append_turn(turn)
            if self.reporter:
                self.reporter.turn_end(task, turn)
            if self.config.stop_on_pass and observation.passed:
                break
            if _should_stop_after_recovery_failure(turn, self.config):
                trace.final_status = "fail"
                break
        if trace.turns and trace.turns[-1].obs_full.get("status") != "pass":
            trace.final_status = "fail"
        return trace


def _format_error_observation(error_type: str) -> Observation:
    return Observation(
        status="fail",
        error_type="format_error",
        obs_for_model=f"status: fail error_type: format_error reason: {error_type}",
        obs_full={"status": "fail", "error_type": "format_error", "reason": error_type},
    )


def _should_recover_submit(
    generation: GenerationResult,
    error_type: str | None,
    config: RunnerConfig,
) -> bool:
    return (
        config.enable_submit_recovery
        and generation.finish_reason == "length"
        and error_type in {"missing_submit", "incomplete_output"}
    )


def _render_recovery_messages(
    task: Task,
    previous_raw_output: str,
    config: RunnerConfig,
) -> list[dict[str, str]]:
    hint = ""
    if config.length_failure_hint:
        hint = "Your previous reasoning exceeded the output budget.\n"
    reasoning_tail = previous_raw_output[-config.recovery_reasoning_tail_chars :].strip()
    return [
        {
            "role": "system",
            "content": (
                "You are a code finalizer. DO NOT THINK. DO NOT EXPLAIN. "
                "Your first token must be <submit>. Output exactly one complete Python solution "
                "inside <submit>...</submit>. Do not output <think>, </think>, <obs>, markdown, "
                "or commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"<input>{task.prompt}</input>\n\n"
                f"{hint}"
                "The previous output was truncated before a valid <submit>. "
                "Use only the brief tail below as context. Do not continue the analysis.\n\n"
                f"<reasoning_tail>{reasoning_tail}</reasoning_tail>\n\n"
                "Now output exactly:\n<submit>complete Python code</submit>"
            ),
        },
    ]


def _should_stop_after_recovery_failure(turn: Turn, config: RunnerConfig) -> bool:
    return (
        config.stop_after_recovery_failure
        and turn.recovery_used
        and turn.obs_full.get("reason") == "missing_submit_after_recovery"
    )
