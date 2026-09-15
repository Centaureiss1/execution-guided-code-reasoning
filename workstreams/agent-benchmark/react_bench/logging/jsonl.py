from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _json_default(value: Any) -> Any:
    if isinstance(value, complex):
        return {
            "__react_bench_type__": "complex",
            "real": value.real,
            "imag": value.imag,
        }
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_object_hook(value: dict[str, Any]) -> Any:
    if value.get("__react_bench_type__") == "complex":
        return complex(value["real"], value["imag"])
    return value


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line, object_hook=_json_object_hook))
    return rows
