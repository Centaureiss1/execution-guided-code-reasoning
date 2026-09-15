from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

from react_bench.agent import ReActRunner
from react_bench.benchmarks import create_benchmark_adapter
from react_bench.config import load_config
from react_bench.distill import distill_failed_prefixes
from react_bench.export import export_sft, select_repair_prefixes
from react_bench.logging import ConsoleReporter, RunLogger
from react_bench.prepare import prepare_kodcode_jsonl, prepare_lcb_jsonl
from react_bench.providers import create_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="react-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_prepare_kodcode_parser(subparsers)
    _add_prepare_lcb_parser(subparsers)
    _add_distill_parser(subparsers)
    _add_export_parser(subparsers)
    _add_select_prefixes_parser(subparsers)
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "prepare-kodcode":
        return _prepare_kodcode(args)
    if args.command == "prepare-lcb":
        return _prepare_lcb(args)
    if args.command == "distill":
        return _distill(args)
    if args.command == "export-sft":
        return _export_sft(args)
    if args.command == "select-prefixes":
        return _select_prefixes(args)
    raise AssertionError(args.command)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run")
    parser.add_argument("--benchmark", required=True, choices=["evalplus", "livecodebench", "kodcode"])
    parser.add_argument("--dataset")
    parser.add_argument("--scenario", default="codegeneration")
    parser.add_argument("--tasks")
    parser.add_argument("--provider", required=True, choices=["azure_openai", "vllm", "openai_compatible", "openai"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env")
    parser.add_argument("--config", default="configs/compact_repair.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--verbose", action="store_true", help="Print per-task and per-turn progress.")
    parser.add_argument("--print-submit", action="store_true", help="Print parsed submit blocks during the run.")
    parser.add_argument("--print-raw", action="store_true", help="Print full raw model outputs during the run.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing runs.jsonl in logs-dir and rerun all tasks.",
    )


def _add_distill_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("distill")
    parser.add_argument("--failed-prefixes", required=True)
    parser.add_argument("--teacher-provider", required=True, choices=["azure_openai", "vllm", "openai_compatible", "openai"])
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env")
    parser.add_argument("--config", default="configs/compact_repair.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--max-turns", type=int, default=3)


def _add_prepare_lcb_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prepare-lcb")
    parser.add_argument("--output", default="data/livecodebench/codegeneration.jsonl")
    parser.add_argument("--version", default="v6")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset-name", default="livecodebench/code_generation_lite")
    parser.add_argument("--subset-file")
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--allow-functional", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")


def _add_prepare_kodcode_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prepare-kodcode")
    parser.add_argument("--output", default="data/kodcode/light_rl_10k.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset-name", default="KodCode/KodCode-Light-RL-10K")
    parser.add_argument("--subset")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--min-gpt-pass-percentage", type=float)
    parser.add_argument("--streaming", action="store_true")


def _add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export-sft")
    parser.add_argument("--teacher-traces", required=True)
    parser.add_argument("--output", required=True)


def _add_select_prefixes_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("select-prefixes")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/compact_repair.yaml")
    parser.add_argument("--min-fail-turns", type=int, default=2)
    parser.add_argument("--max-fail-turns", type=int, default=3)
    parser.add_argument("--allow-format-or-budget", action="store_true")
    parser.add_argument("--no-rejudge", action="store_true")


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config, {"logs_dir": args.logs_dir, "max_turns": args.max_turns})
    adapter = create_benchmark_adapter(
        benchmark=args.benchmark,
        dataset=args.dataset,
        scenario=args.scenario,
        tasks_path=args.tasks,
        limit=args.limit,
    )
    provider = create_provider(args.provider, args.model, args.base_url, args.api_key, args.api_key_env)
    tasks = adapter.load_tasks()
    logger = RunLogger(config.logs_dir)
    task_ids = {task.task_id for task in tasks}
    existing_statuses = (
        {
            task_id: status
            for task_id, status in logger.completed_task_statuses().items()
            if task_id in task_ids
        }
        if not args.no_resume
        else {}
    )
    if existing_statuses:
        tasks_to_run = [task for task in tasks if task.task_id not in existing_statuses]
        existing_passed = sum(status == "pass" for status in existing_statuses.values())
        print(
            "resume: "
            f"skipping={len(existing_statuses)} "
            f"existing_passed={existing_passed} "
            f"remaining={len(tasks_to_run)}",
            flush=True,
        )
    else:
        tasks_to_run = tasks
        existing_passed = 0

    reporter = ConsoleReporter(
        total_tasks=len(tasks),
        verbose=args.verbose or args.print_submit or args.print_raw,
        print_submit=args.print_submit,
        print_raw=args.print_raw,
    )
    runner = ReActRunner(adapter, provider, config, reporter=reporter)
    passed = existing_passed
    for index, task in enumerate(tasks_to_run, start=len(existing_statuses) + 1):
        reporter.task_start(task, index)
        trace = runner.run_task(task)
        logger.log_run(trace)
        passed += trace.final_status == "pass"
        reporter.task_end(task, trace)
    print(
        f"completed={len(existing_statuses) + len(tasks_to_run)} "
        f"passed={passed} "
        f"repair@{config.max_turns}={passed / len(tasks) if tasks else 0:.3f}"
    )
    return 0


def _prepare_lcb(args: argparse.Namespace) -> int:
    written, skipped = prepare_lcb_jsonl(
        output_path=args.output,
        version=args.version,
        split=args.split,
        limit=args.limit,
        dataset_name=args.dataset_name,
        subset_file=args.subset_file,
        stdin_only=not args.allow_functional,
        include_private=args.include_private,
        streaming=not args.no_streaming,
    )
    print(f"wrote={written} skipped={skipped} output={args.output}")
    return 0


def _prepare_kodcode(args: argparse.Namespace) -> int:
    written, skipped = prepare_kodcode_jsonl(
        output_path=args.output,
        split=args.split,
        limit=args.limit,
        dataset_name=args.dataset_name,
        subset=args.subset,
        difficulty=args.difficulty,
        min_gpt_pass_percentage=args.min_gpt_pass_percentage,
        streaming=args.streaming,
    )
    print(f"wrote={written} skipped={skipped} output={args.output}")
    return 0


def _distill(args: argparse.Namespace) -> int:
    config = load_config(args.config, {"logs_dir": args.logs_dir})
    provider = create_provider(
        args.teacher_provider,
        args.teacher_model,
        args.base_url,
        args.api_key,
        args.api_key_env,
    )
    accepted, rejected = distill_failed_prefixes(
        failed_prefixes_path=args.failed_prefixes,
        teacher_provider=provider,
        config=config,
        logs_dir=args.logs_dir,
        max_teacher_turns=args.max_turns,
    )
    print(f"teacher_traces={accepted} rejected_traces={rejected}")
    return 0


def _export_sft(args: argparse.Namespace) -> int:
    count = export_sft(args.teacher_traces, args.output)
    print(f"exported={count} output={args.output}")
    return 0


def _select_prefixes(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    stats = select_repair_prefixes(
        runs_path=args.runs,
        output_path=args.output,
        config=config,
        min_fail_turns=args.min_fail_turns,
        max_fail_turns=args.max_fail_turns,
        clean_coding_only=not args.allow_format_or_budget,
        rejudge=not args.no_rejudge,
    )
    print(
        " ".join(
            [
                f"read={stats['read']}",
                f"selected={stats['selected']}",
                f"skipped_pass_or_too_few_failures={stats['skipped_pass_or_too_few_failures']}",
                f"skipped_too_many_failures={stats['skipped_too_many_failures']}",
                f"skipped_unclean={stats['skipped_unclean']}",
                f"output={args.output}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
