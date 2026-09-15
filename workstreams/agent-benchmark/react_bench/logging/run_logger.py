from __future__ import annotations

from pathlib import Path

from react_bench.logging.jsonl import append_jsonl, read_jsonl
from react_bench.protocol.schema import Trace


class RunLogger:
    def __init__(self, logs_dir: str | Path = "logs") -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.logs_dir / "runs.jsonl"
        self.failed_prefixes_path = self.logs_dir / "failed_prefixes.jsonl"
        self.teacher_traces_path = self.logs_dir / "teacher_traces.jsonl"
        self.rejected_traces_path = self.logs_dir / "rejected_traces.jsonl"

    def log_run(self, trace: Trace) -> None:
        row = trace.to_dict()
        append_jsonl(self.runs_path, row)
        if trace.final_status != "pass":
            append_jsonl(self.failed_prefixes_path, row)

    def completed_task_statuses(self) -> dict[str, str]:
        """Return the last recorded final status for each task in runs.jsonl."""
        if not self.runs_path.exists():
            return {}

        statuses: dict[str, str] = {}
        for row in read_jsonl(self.runs_path):
            try:
                trace = Trace.from_dict(row)
            except Exception:
                continue
            if trace.final_status in {"pass", "fail"}:
                statuses[trace.task.task_id] = trace.final_status
        return statuses

    def log_teacher_trace(self, trace: Trace) -> None:
        append_jsonl(self.teacher_traces_path, trace.to_dict())

    def log_rejected_trace(self, trace: Trace) -> None:
        append_jsonl(self.rejected_traces_path, trace.to_dict())
