from __future__ import annotations

from pathlib import Path
from typing import Any

from react_bench.logging.jsonl import append_jsonl


def prepare_kodcode_jsonl(
    output_path: str,
    split: str = "train",
    limit: int | None = None,
    dataset_name: str = "KodCode/KodCode-Light-RL-10K",
    subset: str | None = None,
    difficulty: str | None = None,
    min_gpt_pass_percentage: float | None = None,
    streaming: bool = False,
) -> tuple[int, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to prepare KodCode JSONL. "
            "Install it with: pip install 'react-bench[prepare]' or pip install datasets"
        ) from exc

    rows = load_dataset(dataset_name, split=split, streaming=streaming)
    output = Path(output_path)
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for row in rows:
        converted = convert_kodcode_row(
            dict(row),
            subset=subset,
            difficulty=difficulty,
            min_gpt_pass_percentage=min_gpt_pass_percentage,
        )
        if converted is None:
            skipped += 1
            continue
        append_jsonl(output, converted)
        written += 1
        if limit is not None and written >= limit:
            break
    return written, skipped


def convert_kodcode_row(
    row: dict[str, Any],
    subset: str | None = None,
    difficulty: str | None = None,
    min_gpt_pass_percentage: float | None = None,
) -> dict[str, Any] | None:
    if subset and str(row.get("subset")) != subset:
        return None
    if difficulty and str(row.get("gpt_difficulty")) != difficulty:
        return None
    if min_gpt_pass_percentage is not None:
        score = row.get("gpt_pass_percentage")
        if score is None or float(score) < min_gpt_pass_percentage:
            return None
    if not str(row.get("question") or "").strip():
        return None
    if not str(row.get("test") or "").strip():
        return None

    return {
        "question_id": str(row.get("question_id") or row.get("id")),
        "question": row.get("question"),
        "solution": row.get("solution"),
        "test": row.get("test"),
        "test_info": row.get("test_info") or [],
        "r1_solution": row.get("r1_solution"),
        "subset": row.get("subset"),
        "gpt_difficulty": row.get("gpt_difficulty"),
        "gpt_pass_percentage": row.get("gpt_pass_percentage"),
        "gpt_pass_trial_num": row.get("gpt_pass_trial_num"),
        "r1_pass_trial_num": row.get("r1_pass_trial_num"),
        "r1_correctness": row.get("r1_correctness"),
        "metadata": row.get("metadata") or {},
    }
