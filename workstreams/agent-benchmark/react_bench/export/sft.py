from __future__ import annotations

from pathlib import Path

from react_bench.logging.jsonl import append_jsonl, read_jsonl
from react_bench.protocol.schema import Trace


def export_sft(teacher_traces_path: str, output_path: str) -> int:
    output = Path(output_path)
    if output.exists():
        output.unlink()
    count = 0
    for row in read_jsonl(teacher_traces_path):
        trace = Trace.from_dict(row)
        if trace.final_status != "pass":
            continue
        append_jsonl(
            output,
            {
                "task_id": trace.task.task_id,
                "benchmark": trace.task.benchmark,
                "dataset": trace.task.dataset,
                "text": trace.to_tagged_text(),
                "metadata": {
                    "run_id": trace.run_id,
                    "provider": trace.provider,
                    "model": trace.model,
                    "turns": len(trace.turns),
                    "feedback_mode": trace.config.get("feedback_mode"),
                },
            },
        )
        count += 1
    return count
