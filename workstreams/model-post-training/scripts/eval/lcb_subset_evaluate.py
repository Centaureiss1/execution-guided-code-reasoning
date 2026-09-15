"""Evaluate a LiveCodeBench custom-output file on a fixed subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lcb_subset_utils import (
    add_lcb_to_path,
    atomic_write_json,
    filter_benchmark_by_subset,
    load_lcb_benchmark,
    summarize_benchmark,
)


def load_custom_code_lists(path: Path, benchmark: list[Any]) -> list[list[str]]:
    with path.open() as f:
        custom_outputs = json.load(f)
    if not isinstance(custom_outputs, list):
        raise SystemExit(f"Custom output must be a JSON list: {path}")
    by_id: dict[str, list[str]] = {}
    for record in custom_outputs:
        if not isinstance(record, dict):
            raise SystemExit("Subset evaluator expects dict custom outputs with question_id/code_list")
        question_id = str(record.get("question_id"))
        code_list = record.get("code_list")
        if not isinstance(code_list, list) or not all(isinstance(item, str) for item in code_list):
            raise SystemExit(f"Invalid code_list for question_id={question_id}")
        by_id[question_id] = code_list

    missing = [str(question.question_id) for question in benchmark if str(question.question_id) not in by_id]
    if missing:
        raise SystemExit(f"Custom output is missing {len(missing)} subset questions. First missing: {missing[:10]}")
    return [by_id[str(question.question_id)] for question in benchmark]


def score_summary(eval_all: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    by_difficulty: dict[str, list[float]] = {}
    by_platform: dict[str, list[float]] = {}
    for result in eval_all:
        pass_1 = float(result["pass@1"])
        by_difficulty.setdefault(result["difficulty"], []).append(pass_1)
        by_platform.setdefault(result["platform"], []).append(pass_1)
    return {
        "num_questions": len(eval_all),
        "num_samples_per_question": len(eval_all[0]["graded_list"]) if eval_all else 0,
        "pass_at_k": {key: value for key, value in metrics.items() if key != "detail"},
        "pass@1_mean": sum(result["pass@1"] for result in eval_all) / len(eval_all),
        "pass@1_by_difficulty": {
            key: sum(values) / len(values) for key, values in sorted(by_difficulty.items())
        },
        "pass@1_by_platform": {
            key: sum(values) / len(values) for key, values in sorted(by_platform.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-output-file", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--lcb-dir", default=None)
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--not-fast", action="store_true")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--streaming-benchmark", action="store_true")
    parser.add_argument("--num-process-evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_lcb_to_path(args.lcb_dir)
    from lcb_runner.evaluation import codegen_metrics, extract_instance_results

    benchmark = load_lcb_benchmark(
        lcb_dir=args.lcb_dir,
        release_version=args.release_version,
        not_fast=args.not_fast,
        start_date=args.start_date,
        end_date=args.end_date,
        streaming=args.streaming_benchmark,
        subset_file=args.subset_file if args.streaming_benchmark else None,
    )
    if not args.streaming_benchmark:
        benchmark = filter_benchmark_by_subset(benchmark, args.subset_file)
    custom_output = Path(args.custom_output_file).resolve()
    generations = load_custom_code_lists(custom_output, benchmark)
    samples = [question.get_evaluation_sample() for question in benchmark]

    metrics, raw_results, metadatas = codegen_metrics(
        samples,
        generations,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )
    graded = extract_instance_results(raw_results)
    eval_all = [
        question.insert_output_evaluation(code_list, code_list, graded_list, metadata=metadata)
        for question, code_list, graded_list, metadata in zip(benchmark, generations, graded, metadatas)
    ]
    save_results = [
        question.insert_output(code_list, code_list)
        for question, code_list in zip(benchmark, generations)
    ]
    prefix = (
        Path(args.output_prefix).resolve()
        if args.output_prefix
        else custom_output.with_suffix("")
    )
    atomic_write_json(prefix.with_name(prefix.name + "_subset_output.json"), save_results)
    atomic_write_json(prefix.with_name(prefix.name + "_subset_eval.json"), [metrics, raw_results, metadatas])
    atomic_write_json(prefix.with_name(prefix.name + "_subset_eval_all.json"), eval_all)
    summary = {
        "custom_output_file": str(custom_output),
        "subset_file": str(Path(args.subset_file).resolve()),
        "benchmark_summary": summarize_benchmark(benchmark),
        "score_summary": score_summary(eval_all, metrics),
    }
    atomic_write_json(prefix.with_name(prefix.name + "_subset_score_summary.json"), summary)
    print(json.dumps(summary["score_summary"], indent=2))
    print(f"Wrote subset eval files with prefix: {prefix}")


if __name__ == "__main__":
    main()
