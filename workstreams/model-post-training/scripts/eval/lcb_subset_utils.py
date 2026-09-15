"""Shared helpers for LiveCodeBench subset generation and evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def add_lcb_to_path(lcb_dir: str | None) -> None:
    if lcb_dir:
        import sys

        resolved = str(Path(lcb_dir).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def load_lcb_benchmark(
    *,
    lcb_dir: str | None,
    release_version: str,
    not_fast: bool,
    start_date: str | None,
    end_date: str | None,
    streaming: bool = False,
    subset_file: str | Path | None = None,
) -> list[Any]:
    add_lcb_to_path(lcb_dir)
    if streaming:
        return load_lcb_benchmark_streaming(
            release_version=release_version,
            not_fast=not_fast,
            start_date=start_date,
            end_date=end_date,
            subset_file=subset_file,
        )

    try:
        from lcb_runner.benchmarks import (
            load_code_generation_dataset,
            load_code_generation_dataset_not_fast,
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not import official LiveCodeBench lcb_runner. "
            "Clone it and pass --lcb-dir /path/to/LiveCodeBench."
        ) from exc

    if not_fast:
        benchmark = load_code_generation_dataset_not_fast(release_version)
        if start_date or end_date:
            print("[WARN] --start-date/--end-date are ignored with --not-fast")
    else:
        benchmark = load_code_generation_dataset(
            release_version,
            start_date=start_date,
            end_date=end_date,
        )
    return sorted(benchmark, key=lambda item: str(item.question_id))


def load_lcb_benchmark_streaming(
    *,
    release_version: str,
    not_fast: bool,
    start_date: str | None,
    end_date: str | None,
    subset_file: str | Path | None = None,
) -> list[Any]:
    try:
        from datasets import load_dataset
        from lcb_runner.benchmarks import CodeGenerationProblem
    except ImportError as exc:
        raise SystemExit("Streaming LCB load requires datasets and lcb_runner imports") from exc

    subset_ids = set(load_subset_question_ids(subset_file)) if subset_file else None
    dataset_name = "livecodebench/code_generation" if not_fast else "livecodebench/code_generation_lite"
    kwargs: dict[str, Any] = {
        "split": "test",
        "trust_remote_code": True,
        "streaming": True,
    }
    if not not_fast:
        kwargs["version_tag"] = release_version

    p_start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    p_end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    benchmark: list[Any] = []
    for row in load_dataset(dataset_name, **kwargs):
        question_id = str(row["question_id"])
        if subset_ids is not None and question_id not in subset_ids:
            continue
        question = CodeGenerationProblem(**row)
        if p_start_date and question.contest_date < p_start_date:
            continue
        if p_end_date and question.contest_date > p_end_date:
            continue
        benchmark.append(question)
        if subset_ids is not None and len(benchmark) == len(subset_ids):
            break
    benchmark = sorted(benchmark, key=lambda item: str(item.question_id))
    print(f"Loaded {len(benchmark)} problems with streaming=True")
    return benchmark


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def item_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def date_bin(date_value: Any) -> str:
    if isinstance(date_value, datetime):
        dt = date_value
    else:
        dt = datetime.fromisoformat(str(date_value))
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{quarter}"


def has_starter_code(question: Any) -> bool:
    return bool(str(getattr(question, "starter_code", "") or "").strip())


def question_summary(question: Any) -> dict[str, Any]:
    return {
        "question_id": str(question.question_id),
        "question_title": str(getattr(question, "question_title", "")),
        "platform": item_value(question.platform),
        "difficulty": item_value(question.difficulty),
        "contest_date": question.contest_date.isoformat(),
        "date_bin": date_bin(question.contest_date),
        "has_starter_code": has_starter_code(question),
    }


def summarize_benchmark(benchmark: list[Any]) -> dict[str, Any]:
    records = [question_summary(question) for question in benchmark]
    return {
        "num_questions": len(records),
        "platform": dict(Counter(record["platform"] for record in records)),
        "difficulty": dict(Counter(record["difficulty"] for record in records)),
        "date_bin": dict(Counter(record["date_bin"] for record in records)),
        "starter_code": dict(
            Counter("with_starter" if record["has_starter_code"] else "no_starter" for record in records)
        ),
        "platform_difficulty": dict(
            Counter(f"{record['platform']}|{record['difficulty']}" for record in records)
        ),
    }


def load_subset_question_ids(path: str | Path) -> list[str]:
    subset_path = Path(path)
    with subset_path.open() as f:
        data = json.load(f)
    ids = data.get("question_ids") if isinstance(data, dict) else data
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise SystemExit(f"Invalid subset file; expected question_ids list: {subset_path}")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"Subset file contains duplicate question_ids: {subset_path}")
    return ids


def filter_benchmark_by_subset(benchmark: list[Any], subset_file: str | Path) -> list[Any]:
    question_ids = load_subset_question_ids(subset_file)
    by_id = {str(question.question_id): question for question in benchmark}
    missing = [question_id for question_id in question_ids if question_id not in by_id]
    if missing:
        raise SystemExit(
            f"Subset file has {len(missing)} question_ids not found in selected benchmark. "
            f"First missing ids: {missing[:10]}"
        )
    return [by_id[question_id] for question_id in question_ids]
