"""Create a fixed, diverse LiveCodeBench code-generation subset."""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lcb_subset_utils import (
    add_lcb_to_path,
    atomic_write_json,
    date_bin,
    has_starter_code,
    item_value,
    load_lcb_benchmark,
    question_summary,
    summarize_benchmark,
)


def group_key(question: Any) -> tuple[str, str]:
    return item_value(question.platform), item_value(question.difficulty)


def subgroup_key(question: Any) -> tuple[str, str]:
    starter = "with_starter" if has_starter_code(question) else "no_starter"
    return date_bin(question.contest_date), starter


def allocate_quotas(groups: dict[tuple[str, str], list[Any]], total: int) -> dict[tuple[str, str], int]:
    nonempty = {key: value for key, value in groups.items() if value}
    if total < len(nonempty):
        raise SystemExit(f"Cannot allocate {total} examples across {len(nonempty)} non-empty groups")

    full_size = sum(len(items) for items in nonempty.values())
    quotas: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, int, tuple[str, str]]] = []
    for key, items in nonempty.items():
        ideal = total * len(items) / full_size
        quota = max(1, min(len(items), math.floor(ideal)))
        quotas[key] = quota
        remainders.append((ideal - math.floor(ideal), len(items), key))

    while sum(quotas.values()) < total:
        for _, _, key in sorted(remainders, reverse=True):
            if sum(quotas.values()) >= total:
                break
            if quotas[key] < len(nonempty[key]):
                quotas[key] += 1

    while sum(quotas.values()) > total:
        for _, _, key in sorted(remainders):
            if sum(quotas.values()) <= total:
                break
            if quotas[key] > 1:
                quotas[key] -= 1

    return quotas


def select_within_group(items: list[Any], quota: int, rng: random.Random) -> list[Any]:
    buckets: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in sorted(items, key=lambda question: str(question.question_id)):
        buckets[subgroup_key(item)].append(item)
    for bucket_items in buckets.values():
        rng.shuffle(bucket_items)

    selected: list[Any] = []
    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)
    while len(selected) < quota and bucket_keys:
        next_keys: list[tuple[str, str]] = []
        for key in bucket_keys:
            if len(selected) >= quota:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        bucket_keys = next_keys
    return selected


def select_subset(benchmark: list[Any], num_questions: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for question in benchmark:
        groups[group_key(question)].append(question)

    quotas = allocate_quotas(groups, num_questions)
    selected: list[Any] = []
    for key in sorted(quotas):
        selected.extend(select_within_group(groups[key], quotas[key], rng))

    if len(selected) != num_questions:
        raise SystemExit(f"Internal error: selected {len(selected)} != {num_questions}")
    return sorted(selected, key=lambda question: str(question.question_id))


def load_lcb_metadata_streaming(args: argparse.Namespace) -> list[Any]:
    add_lcb_to_path(args.lcb_dir)
    from datasets import load_dataset

    dataset_name = "livecodebench/code_generation" if args.not_fast else "livecodebench/code_generation_lite"
    kwargs: dict[str, Any] = {
        "split": "test",
        "trust_remote_code": True,
        "streaming": True,
    }
    if not args.not_fast:
        kwargs["version_tag"] = args.release_version

    p_start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    p_end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None
    records: list[Any] = []
    for row in load_dataset(dataset_name, **kwargs):
        contest_date = datetime.fromisoformat(str(row["contest_date"]))
        if p_start_date and contest_date < p_start_date:
            continue
        if p_end_date and contest_date > p_end_date:
            continue
        records.append(
            SimpleNamespace(
                question_id=str(row["question_id"]),
                question_title=str(row.get("question_title", "")),
                platform=str(row["platform"]),
                difficulty=str(row["difficulty"]),
                contest_date=contest_date,
                starter_code=str(row.get("starter_code", "") or ""),
            )
        )
    records = sorted(records, key=lambda question: str(question.question_id))
    print(f"Loaded {len(records)} metadata rows with streaming=True")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcb-dir", default=None)
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--not-fast", action="store_true")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--no-streaming", action="store_true", help="Use the normal HF datasets cache loader")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_streaming:
        benchmark = load_lcb_benchmark(
            lcb_dir=args.lcb_dir,
            release_version=args.release_version,
            not_fast=args.not_fast,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        benchmark = load_lcb_metadata_streaming(args)
    selected = select_subset(benchmark, args.num_questions, args.seed)
    source_name = "livecodebench/code_generation" if args.not_fast else "livecodebench/code_generation_lite"
    output = {
        "release_version": args.release_version,
        "source": source_name,
        "not_fast": args.not_fast,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "seed": args.seed,
        "num_questions": len(selected),
        "selection": "platform+difficulty proportional; date_bin+starter_code round-robin within groups",
        "question_ids": [str(question.question_id) for question in selected],
        "summary": {
            "full": summarize_benchmark(benchmark),
            "subset": summarize_benchmark(selected),
        },
        "questions": [question_summary(question) for question in selected],
    }
    atomic_write_json(Path(args.output).resolve(), output)
    print(f"Wrote {len(selected)} question ids to {Path(args.output).resolve()}")
    print("Subset summary:")
    for key, value in output["summary"]["subset"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
