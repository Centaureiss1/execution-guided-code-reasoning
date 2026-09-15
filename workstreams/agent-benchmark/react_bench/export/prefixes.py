from __future__ import annotations

from dataclasses import replace
from typing import Any

from react_bench.benchmarks.factory import create_benchmark_adapter
from react_bench.config import RunnerConfig
from react_bench.logging.jsonl import read_jsonl, write_jsonl
from react_bench.protocol.schema import Trace


CODING_ERROR_TYPES = {"wrong_answer", "runtime_error", "syntax_error"}


def select_repair_prefixes(
    runs_path: str,
    output_path: str,
    config: RunnerConfig,
    min_fail_turns: int = 2,
    max_fail_turns: int = 3,
    clean_coding_only: bool = True,
    rejudge: bool = True,
) -> dict[str, int]:
    selected = []
    stats = {
        "read": 0,
        "selected": 0,
        "skipped_pass_or_too_few_failures": 0,
        "skipped_too_many_failures": 0,
        "skipped_unclean": 0,
    }
    for row in read_jsonl(runs_path):
        stats["read"] += 1
        trace = Trace.from_dict(row)
        classified = _classify_turns(trace, config, rejudge=rejudge)
        failed_turns = [item for item in classified if item["status"] == "fail"]
        fail_count = len(failed_turns)
        if any(item["status"] != "fail" for item in classified[:fail_count]):
            stats["skipped_pass_or_too_few_failures"] += 1
            continue
        if fail_count < min_fail_turns:
            stats["skipped_pass_or_too_few_failures"] += 1
            continue
        if fail_count > max_fail_turns:
            stats["skipped_too_many_failures"] += 1
            continue
        if clean_coding_only and not _clean_coding_failures(failed_turns):
            stats["skipped_unclean"] += 1
            continue

        prefix_len = max(1, fail_count - 1)
        prefix = _slice_trace(trace, classified[:prefix_len])
        selected.append(prefix.to_dict())
        stats["selected"] += 1

    write_jsonl(output_path, selected)
    return stats


def _classify_turns(trace: Trace, config: RunnerConfig, rejudge: bool) -> list[dict[str, Any]]:
    adapter = None
    if rejudge:
        adapter = create_benchmark_adapter(
            benchmark=trace.task.benchmark,
            dataset=trace.task.dataset,
            scenario=trace.task.scenario,
        )
    classified = []
    for turn in trace.turns:
        status = str(turn.obs_full.get("status") or "fail")
        error_type = turn.obs_full.get("error_type")
        obs_for_model = turn.obs_for_model
        if rejudge and adapter is not None and turn.parsed_submit:
            obs = adapter.judge(trace.task, turn.parsed_submit, config)
            status = obs.status
            error_type = obs.error_type
            obs_for_model = obs.obs_for_model
        elif not turn.parsed_submit:
            status = "fail"
            error_type = "format_error"
        classified.append(
            {
                "turn": turn,
                "status": status,
                "error_type": error_type,
                "obs_for_model": obs_for_model,
            }
        )
    return classified


def _clean_coding_failures(failed_turns: list[dict[str, Any]]) -> bool:
    for item in failed_turns:
        turn = item["turn"]
        if turn.parse_status != "ok" or not turn.parsed_submit:
            return False
        if turn.finish_reason == "length" or turn.recovery_used:
            return False
        if item["error_type"] not in CODING_ERROR_TYPES:
            return False
    return True


def _slice_trace(trace: Trace, classified_prefix: list[dict[str, Any]]) -> Trace:
    prefix = Trace(
        task=trace.task,
        provider=trace.provider,
        model=trace.model,
        config=trace.config,
        run_id=trace.run_id,
        created_at=trace.created_at,
    )
    prefix.turns = [
        replace(
            item["turn"],
            obs_for_model=item["obs_for_model"],
            obs_full={
                **item["turn"].obs_full,
                "status": item["status"],
                "error_type": item["error_type"],
            },
        )
        for item in classified_prefix
    ]
    prefix.final_status = prefix.turns[-1].obs_full.get("status", "fail") if prefix.turns else "fail"
    return prefix
