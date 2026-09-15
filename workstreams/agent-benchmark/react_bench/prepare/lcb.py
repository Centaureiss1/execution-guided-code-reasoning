from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from react_bench.benchmarks.livecodebench import coerce_lcb_cases
from react_bench.logging.jsonl import append_jsonl


def prepare_lcb_jsonl(
    output_path: str,
    version: str = "v6",
    split: str = "test",
    limit: int | None = None,
    dataset_name: str = "livecodebench/code_generation_lite",
    subset_file: str | None = None,
    stdin_only: bool = True,
    include_private: bool = False,
    streaming: bool = True,
) -> tuple[int, int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to prepare LiveCodeBench JSONL. "
            "Install it with: pip install 'react-bench[prepare]' or pip install datasets"
        ) from exc

    subset_ids = _load_subset_ids(subset_file)
    rows = _load_lcb_dataset(load_dataset, dataset_name, version, split, streaming)
    output = Path(output_path)
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    if subset_ids is not None:
        return _write_subset_rows(
            rows=rows,
            subset_ids=subset_ids[:limit] if limit is not None else subset_ids,
            output=output,
            stdin_only=stdin_only,
            include_private=include_private,
        )

    written = 0
    skipped = 0
    for row in rows:
        converted = convert_lcb_row(row, stdin_only=stdin_only, include_private=include_private)
        if converted is None:
            skipped += 1
            continue
        append_jsonl(output, converted)
        written += 1
        if limit is not None and written >= limit:
            break
    return written, skipped


def convert_lcb_row(
    row: dict[str, Any],
    stdin_only: bool = True,
    include_private: bool = False,
) -> dict[str, Any] | None:
    public_cases = coerce_lcb_cases(row.get("public_test_cases"))
    private_cases = coerce_lcb_cases(row.get("private_test_cases"))
    if stdin_only:
        public_cases = [case for case in public_cases if _is_stdio_case(case)]
        private_cases = [case for case in private_cases if _is_stdio_case(case)]
    if not public_cases:
        return None

    prompt = str(row.get("question_content") or "")
    starter_code = str(row.get("starter_code") or "").strip()
    if starter_code:
        prompt = prompt.rstrip() + "\n\nStarter code:\n" + starter_code

    converted = {
        "question_id": str(row.get("question_id") or row.get("task_id") or row.get("id")),
        "question_title": row.get("question_title"),
        "question_content": prompt,
        "platform": row.get("platform"),
        "contest_id": row.get("contest_id"),
        "contest_date": row.get("contest_date"),
        "difficulty": row.get("difficulty"),
        "public_test_cases": [
            {
                "input": case.get("input", ""),
                "output": case.get("output", ""),
                "testtype": case.get("testtype") or case.get("test_type") or case.get("type"),
            }
            for case in public_cases
        ],
        "metadata": _maybe_json(row.get("metadata")) or {},
    }
    if include_private:
        converted["private_test_cases"] = [
            {
                "input": case.get("input", ""),
                "output": case.get("output", ""),
                "testtype": case.get("testtype") or case.get("test_type") or case.get("type"),
            }
            for case in private_cases
        ]
    return converted


def _load_lcb_dataset(
    load_dataset: Any,
    dataset_name: str,
    version: str,
    split: str,
    streaming: bool,
) -> Iterable[dict[str, Any]]:
    try:
        return load_dataset(dataset_name, version, split=split, streaming=streaming)
    except Exception as first_exc:
        try:
            return load_dataset(
                dataset_name,
                split=split,
                version_tag=version,
                trust_remote_code=True,
                streaming=streaming,
            )
        except Exception as second_exc:
            raise RuntimeError(
                f"Could not load {dataset_name} version={version!r}. "
                "Try --version v6 for a smaller recent slice or --version release_v6 for the full release."
            ) from second_exc


def _load_subset_ids(subset_file: str | None) -> list[str] | None:
    if subset_file is None:
        return None

    data = json.loads(Path(subset_file).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        ids = data.get("question_ids")
        if ids is None and isinstance(data.get("questions"), list):
            ids = [item.get("question_id") for item in data["questions"]]
    elif isinstance(data, list):
        ids = [
            item.get("question_id") if isinstance(item, dict) else item
            for item in data
        ]
    else:
        ids = None

    if not isinstance(ids, list) or not ids:
        raise ValueError(f"Could not find question_ids in subset file: {subset_file}")
    return [str(item) for item in ids if str(item).strip()]


def _write_subset_rows(
    rows: Iterable[dict[str, Any]],
    subset_ids: list[str],
    output: Path,
    stdin_only: bool,
    include_private: bool,
) -> tuple[int, int]:
    wanted = set(subset_ids)
    converted_by_id: dict[str, dict[str, Any]] = {}
    found_ids: set[str] = set()
    seen = 0
    skipped = 0

    for row in rows:
        question_id = str(row.get("question_id") or row.get("task_id") or row.get("id"))
        if question_id not in wanted:
            continue
        seen += 1
        found_ids.add(question_id)
        converted = convert_lcb_row(
            row,
            stdin_only=stdin_only,
            include_private=include_private,
        )
        if converted is None:
            skipped += 1
        else:
            converted_by_id[question_id] = converted
        if seen >= len(wanted):
            break

    written = 0
    for question_id in subset_ids:
        converted = converted_by_id.get(question_id)
        if converted is None:
            continue
        append_jsonl(output, converted)
        written += 1

    skipped += len(wanted - found_ids)
    return written, skipped


def _is_stdio_case(case: dict[str, Any]) -> bool:
    testtype = str(case.get("testtype") or case.get("test_type") or case.get("type") or "stdin").lower()
    return testtype in {"stdin", "standard_input", "standard input", "io", "input_output"}


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
