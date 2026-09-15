"""Generate LiveCodeBench custom outputs with Qwen3/vLLM.

This script intentionally delegates dataset loading and scoring compatibility to
the official LiveCodeBench package. It only owns Qwen3 chat-template prompting,
vLLM generation, and lightweight code extraction into the custom evaluator
format:

    [{"question_id": "...", "code_list": ["..."], "raw_output_list": ["..."]}]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from lcb_subset_utils import filter_benchmark_by_subset, load_lcb_benchmark as load_lcb_benchmark_shared


SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)

FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)

FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def add_lcb_to_path(lcb_dir: str | None) -> None:
    if lcb_dir:
        sys.path.insert(0, str(Path(lcb_dir).resolve()))


def load_lcb_benchmark(
    *,
    lcb_dir: str | None,
    release_version: str,
    not_fast: bool,
    start_date: str | None,
    end_date: str | None,
) -> list[Any]:
    add_lcb_to_path(lcb_dir)
    try:
        from lcb_runner.benchmarks import (
            load_code_generation_dataset,
            load_code_generation_dataset_not_fast,
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not import official LiveCodeBench lcb_runner.\n"
            "Clone it and pass --lcb-dir, for example:\n"
            "  git clone https://github.com/LiveCodeBench/LiveCodeBench.git "
            "/tmp/cs639_grpo/LiveCodeBench\n"
            "  --lcb-dir /tmp/cs639_grpo/LiveCodeBench"
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


def build_lcb_user_prompt(question: Any) -> str:
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"### Format: {FORMAT_WITH_STARTER}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMAT_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def resolve_enable_thinking(mode: str) -> bool | None:
    if mode == "thinking":
        return True
    if mode == "auto":
        return None
    if mode == "no_thinking":
        return False
    raise ValueError(f"Unknown thinking mode: {mode}")


def build_prompt_text(tokenizer: Any, question: Any, thinking_mode: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_lcb_user_prompt(question)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=resolve_enable_thinking(thinking_mode),
    )


def strip_think(text: str) -> str:
    if "</think>" in text:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text.strip()


def clean_code_block(block: str) -> str:
    block = block.strip()
    block = re.sub(r"(?i)^python\s*\n", "", block).strip()
    return block.replace("\r\n", "\n")


def extract_code(text: str, *, may_think: bool) -> str:
    candidate = strip_think(text) if may_think else text.strip()

    blocks = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)(?:```|$)",
        candidate,
        flags=re.DOTALL | re.IGNORECASE,
    )
    blocks = [clean_code_block(block) for block in blocks if block.strip()]
    if blocks:
        # LiveCodeBench prompts request the final answer in a code fence. If the
        # model emits reasoning snippets first, the last fenced block is usually
        # the submission.
        return blocks[-1]

    starts = []
    for pattern in (
        r"(?m)^[ \t]*(?:from\s+\S+\s+import\s+\S+|import\s+\S+)",
        r"(?m)^[ \t]*(?:class|def)\s+[A-Za-z_]\w*",
        r"(?m)^[ \t]*if\s+__name__\s*==",
    ):
        starts.extend(match.start() for match in re.finditer(pattern, candidate))
    if starts:
        return candidate[min(starts):].strip()

    return candidate


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Existing output is not a JSON list: {path}")
    return data


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


def parse_stop(stop: str) -> list[str] | None:
    if not stop:
        return None
    values = [item for item in stop.split(",") if item]
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Merged/full model directory")
    parser.add_argument("--output", required=True, help="LCB custom output JSON")
    parser.add_argument("--lcb-dir", default=os.environ.get("LCB_DIR"))
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--not-fast", action="store_true", help="Use full code_generation instead of lite")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--subset-file", default=None, help="JSON file with question_ids to evaluate")
    parser.add_argument("--streaming-benchmark", action="store_true", help="Stream LCB rows instead of building an Arrow cache")
    parser.add_argument("--thinking-mode", choices=["thinking", "auto", "no_thinking"], default="thinking")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--stop", default="###", help="Comma-separated stop strings; empty disables stops")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N problems")
    parser.add_argument("--debug", action="store_true", help="Alias for --limit 15 when --limit is absent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.streaming_benchmark:
        benchmark = load_lcb_benchmark_shared(
            lcb_dir=args.lcb_dir,
            release_version=args.release_version,
            not_fast=args.not_fast,
            start_date=args.start_date,
            end_date=args.end_date,
            streaming=True,
            subset_file=args.subset_file,
        )
    else:
        benchmark = load_lcb_benchmark(
            lcb_dir=args.lcb_dir,
            release_version=args.release_version,
            not_fast=args.not_fast,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    if args.subset_file and not args.streaming_benchmark:
        benchmark = filter_benchmark_by_subset(benchmark, args.subset_file)
    if args.debug and args.limit is None:
        args.limit = 15
    if args.limit is not None:
        benchmark = benchmark[: args.limit]

    output_path = Path(args.output).resolve()
    records = load_existing(output_path)
    by_question_id = {str(record["question_id"]): record for record in records}
    remaining = [
        question for question in benchmark if str(question.question_id) not in by_question_id
    ]

    print(
        f"Loaded {len(benchmark)} target problems; "
        f"{len(by_question_id)} already generated; {len(remaining)} remaining."
    )
    if not remaining:
        ordered = [by_question_id[str(q.question_id)] for q in benchmark]
        atomic_write_json(output_path, ordered)
        print(f"All outputs already present: {output_path}")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="left",
    )

    print(f"Loading vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        max_num_seqs=10,
    )
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        stop=parse_stop(args.stop),
    )

    may_think = args.thinking_mode != "no_thinking"
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start : start + args.batch_size]
        prompts = [
            build_prompt_text(tokenizer, question, args.thinking_mode)
            for question in batch
        ]
        print(
            f"Generating batch {start // args.batch_size + 1}: "
            f"{len(batch)} problems"
        )
        outputs = llm.generate(prompts, sampling)

        for question, output in zip(batch, outputs):
            raw_outputs = [item.text for item in output.outputs]
            code_list = [
                extract_code(raw_output, may_think=may_think)
                for raw_output in raw_outputs
            ]
            by_question_id[str(question.question_id)] = {
                "question_id": str(question.question_id),
                "code_list": code_list,
                "raw_output_list": raw_outputs,
            }

        ordered = [
            by_question_id[str(question.question_id)]
            for question in benchmark
            if str(question.question_id) in by_question_id
        ]
        atomic_write_json(output_path, ordered)
        print(f"Saved {len(ordered)} records to {output_path}")

    print(f"Done. Custom output: {output_path}")


if __name__ == "__main__":
    main()
