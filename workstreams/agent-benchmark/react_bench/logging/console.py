from __future__ import annotations

from react_bench.protocol.render import estimate_tokens
from react_bench.protocol.schema import Task, Trace, Turn


class ConsoleReporter:
    def __init__(
        self,
        total_tasks: int,
        verbose: bool = False,
        print_submit: bool = False,
        print_raw: bool = False,
    ) -> None:
        self.total_tasks = total_tasks
        self.verbose = verbose
        self.print_submit = print_submit
        self.print_raw = print_raw
        self.current_index = 0

    def task_start(self, task: Task, index: int) -> None:
        self.current_index = index
        if self.verbose:
            print(f"[{index}/{self.total_tasks}] {task.benchmark}:{task.task_id} start", flush=True)

    def turn_start(self, task: Task, turn_index: int) -> None:
        if self.verbose:
            print(f"  turn={turn_index} generate...", flush=True)

    def turn_end(self, task: Task, turn: Turn) -> None:
        if not self.verbose:
            return
        submit_present = "yes" if turn.parsed_submit else "no"
        think_tokens = estimate_tokens(turn.parsed_think)
        submit_tokens = estimate_tokens(turn.parsed_submit)
        prompt_tokens = turn.token_usage.get("prompt_tokens")
        completion_tokens = turn.token_usage.get("completion_tokens")
        usage = ""
        if prompt_tokens is not None or completion_tokens is not None:
            usage = f" prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
        latency = f"{turn.latency:.2f}s" if turn.latency is not None else "n/a"
        print(
            "  "
            f"turn={turn.turn_index} parse={turn.parse_status} submit={submit_present} "
            f"think_tokens~={think_tokens} submit_tokens~={submit_tokens} "
            f"finish={turn.finish_reason} latency={latency}{usage}",
            flush=True,
        )
        if turn.recovery_used:
            recovery_prompt_tokens = turn.recovery_token_usage.get("prompt_tokens")
            recovery_completion_tokens = turn.recovery_token_usage.get("completion_tokens")
            print(
                "  "
                f"turn={turn.turn_index} recovery=parse={turn.recovery_parse_status} "
                f"finish={turn.recovery_finish_reason} "
                f"disable_thinking={turn.recovery_disable_thinking} "
                f"prompt_tokens={recovery_prompt_tokens} completion_tokens={recovery_completion_tokens}",
                flush=True,
            )
        print(f"  turn={turn.turn_index} obs={turn.obs_for_model}", flush=True)
        if self.print_submit and turn.parsed_submit:
            print(f"  turn={turn.turn_index} parsed_submit:\n{turn.parsed_submit}", flush=True)
        if self.print_raw:
            print(f"  turn={turn.turn_index} raw_output:\n{turn.raw_output}", flush=True)

    def task_end(self, task: Task, trace: Trace) -> None:
        print(
            f"[{self.current_index}/{self.total_tasks}] {task.task_id}: "
            f"{trace.final_status} turns={len(trace.turns)}",
            flush=True,
        )
