"""Score and tier the Qwen3-8B GRPO pool.

This script is the 8B direct-GRPO counterpart to score_and_tier.py.

Pipeline:
  1. Load data/processed/grpo_pool_qwen3_8b.jsonl.
  2. Run Qwen3-8B base with 8 rollouts per problem.
  3. Run Qwen3-8B with the configured thinking mode and sampling settings.
  4. Execute each completion against the problem asserts.
  5. Save resumable scored rows with pass_count and pass_rate.
  6. Save all eligible hard data by default:
       pass_count in {1, 2} out of 8.
     This matches the 1.7B hard/Main setting while leaving source-quota
     sampling (for example KodCode 1k + OpenCode 4k) to a later step.

Output:
  data/processed/grpo_pool_qwen3_8b_scored.jsonl
  data/processed/grpo_train_hard_qwen3_8b.jsonl
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "Qwen3-8B"
DEFAULT_POOL_PATH = ROOT / "data" / "processed" / "grpo_pool_qwen3_8b.jsonl"
DEFAULT_SCORED_PATH = (
    ROOT / "data" / "processed" / "grpo_pool_qwen3_8b_scored.jsonl"
)
DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "grpo_train_hard_qwen3_8b.jsonl"

QWEN3_CODE_SYSTEM = (
    "You are an expert Python programmer.\n"
    "Solve the task using exactly this final-answer format:\n\n"
    "<think>\n"
    "reasoning\n"
    "</think>\n\n"
    "```python\n"
    "complete executable Python code\n"
    "```"
)

SYSTEM = QWEN3_CODE_SYSTEM
TRAIN_SYSTEM = QWEN3_CODE_SYSTEM

DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 8
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_BATCH_SIZE = 64
DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
DEFAULT_TEST_TIMEOUT = 2
DEFAULT_TEST_WORKERS = 1
DEFAULT_PROGRESS_EVERY = 1

PYTHON_BLOCK_RE = re.compile(
    r"```python\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
GENERIC_BLOCK_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)
CODE_START_RE = re.compile(
    r"(?m)^((?:from\s+\S+\s+import\s+|import\s+\S+|class\s+\w+|def\s+\w+).*)",
    re.DOTALL,
)
DEFAULT_QUESTION_PREVIEW_CHARS = 120

PASS_COUNTS = [1, 2]
SOURCE_TARGETS = {
    "kodcode": 1000,
    "opencode": 4000,
}
BUILTIN_NAMES = set(dir(builtins)) | {"True", "False", "None"}
KNOWN_EXTERNAL_NAMES = {
    "collections",
    "datetime",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "np",
    "numpy",
    "operator",
    "os",
    "pd",
    "pandas",
    "pathlib",
    "pytest",
    "random",
    "re",
    "sqlite3",
    "statistics",
    "sys",
    "tempfile",
    "unittest",
}


def log(message: str) -> None:
    print(message, flush=True)


def log_step(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log(f"[{timestamp}] {message}")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def parse_pass_counts(value: str) -> set[int]:
    pass_counts: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            pass_counts.add(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid pass_count value {item!r}; expected comma-separated integers."
            ) from exc
    if not pass_counts:
        raise argparse.ArgumentTypeError("At least one pass_count must be provided.")
    return pass_counts


def record_uid(rec: dict) -> str:
    uid = str(rec.get("uid") or "").strip()
    if uid:
        return uid
    source = str(rec.get("source") or "unknown").strip()
    qid = str(rec.get("question_id") or rec.get("fingerprint") or "").strip()
    return f"{source}:{qid}"


def question_preview(rec: dict, max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    content = ""
    for msg in rec.get("messages", []):
        if msg.get("role") == "user":
            content = str(msg.get("content") or "")
            break

    content = re.sub(r"\s+", " ", content).strip()
    if len(content) <= max_chars:
        return content
    return content[: max_chars - 3].rstrip() + "..."


def defined_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def call_root_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        value = func.value
        while isinstance(value, ast.Attribute):
            value = value.value
        if isinstance(value, ast.Name):
            return value.id
    return None


def infer_required_symbols(tests: list[str]) -> list[str]:
    counts: dict[str, int] = {}

    for test in tests:
        try:
            tree = ast.parse(test)
        except SyntaxError:
            continue

        local_names = defined_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = call_root_name(node.func)
            if not name:
                continue
            if (
                name in local_names
                or name in BUILTIN_NAMES
                or name in KNOWN_EXTERNAL_NAMES
            ):
                continue
            counts[name] = counts.get(name, 0) + 1

    return [
        name
        for name, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def add_symbol_contract(content: str, tests: list[str]) -> str:
    names = infer_required_symbols(tests)
    if not names or "Your solution must define" in content:
        return content

    joined = ", ".join(f"`{name}`" for name in names)
    contract = f"Your solution must define these top-level names: {joined}."
    return f"{content.rstrip()}\n\n{contract}"


def messages_with_test_contract(rec: dict, system_prompt_mode: str = "simple") -> list[dict]:
    tests = [str(test) for test in rec.get("tests", [])]
    messages: list[dict] = []
    user_contract_added = False

    for msg in rec.get("messages", []):
        copied = dict(msg)
        if copied.get("role") == "system":
            if system_prompt_mode == "simple":
                copied["content"] = SYSTEM
            elif system_prompt_mode == "train":
                copied["content"] = TRAIN_SYSTEM
            elif system_prompt_mode == "preserve":
                copied["content"] = str(copied.get("content") or SYSTEM)
            else:
                raise ValueError(f"Unsupported system_prompt_mode: {system_prompt_mode}")
        elif copied.get("role") == "user" and not user_contract_added:
            copied["content"] = add_symbol_contract(
                str(copied.get("content") or ""),
                tests,
            )
            user_contract_added = True
        messages.append(copied)

    return messages


def training_messages(row: dict) -> list[dict]:
    messages: list[dict] = []
    system_replaced = False

    for msg in row.get("messages", []):
        copied = dict(msg)
        if copied.get("role") == "system" and not system_replaced:
            copied["content"] = TRAIN_SYSTEM
            system_replaced = True
        messages.append(copied)

    if not system_replaced:
        messages.insert(0, {"role": "system", "content": TRAIN_SYSTEM})

    return messages


def format_indices(indices: list[int], max_items: int = 8) -> str:
    if not indices:
        return "-"
    shown = ",".join(str(idx) for idx in indices[:max_items])
    if len(indices) > max_items:
        shown += ",..."
    return shown


def extract_code(text: str) -> str | None:
    tail = text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()

    match = PYTHON_BLOCK_RE.search(tail)
    if match:
        code = match.group(1).strip()
        return code or None

    match = GENERIC_BLOCK_RE.search(tail)
    if match:
        code = match.group(1).strip()
        if re.search(r"\b(def|class)\s+\w+|\bimport\s+\S+|\bfrom\s+\S+\s+import\b", code):
            return code

    match = CODE_START_RE.search(tail)
    if match:
        code = textwrap.dedent(match.group(1)).strip()
        return code or None
    return None


def run_tests(code: str, tests: list[str], timeout: int) -> bool:
    if not tests:
        return False

    program = code + "\n\n" + "\n".join(tests)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.Popen(
                [sys.executable, "-c", program],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=tmpdir,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                return False
        return proc.returncode == 0
    except Exception:
        return False


def score_completions(
    completions: list[str],
    tests: list[str],
    timeout: int,
    *,
    rollout_progress: bool = False,
    record_label: str = "",
) -> tuple[int, list[int]]:
    passed_indices: list[int] = []
    for idx, completion in enumerate(completions):
        rollout_time = time.time()
        code = extract_code(completion)
        passed = False
        if code:
            passed = run_tests(code, tests, timeout=timeout)
            if passed:
                passed_indices.append(idx)
        if rollout_progress:
            if not code:
                status = "no_code"
            elif passed:
                status = "pass"
            else:
                status = "fail"
            code_chars = len(code) if code else 0
            label = f" {record_label}" if record_label else ""
            log(
                f"    rollout {idx + 1:>2}/{len(completions):<2}"
                f"{label} status={status:<7} "
                f"output_chars={len(completion):>5} "
                f"code_chars={code_chars:>5} "
                f"time={time.time() - rollout_time:.2f}s"
            )
    return len(passed_indices), passed_indices


def score_rollout_job(
    rec_idx: int,
    rollout_idx: int,
    completion: str,
    tests: list[str],
    timeout: int,
) -> dict[str, Any]:
    rollout_time = time.time()
    code = extract_code(completion)
    if not code:
        return {
            "rec_idx": rec_idx,
            "rollout_idx": rollout_idx,
            "passed": False,
            "status": "no_code",
            "output_chars": len(completion),
            "code_chars": 0,
            "elapsed": time.time() - rollout_time,
        }

    passed = run_tests(code, tests, timeout=timeout)
    return {
        "rec_idx": rec_idx,
        "rollout_idx": rollout_idx,
        "passed": passed,
        "status": "pass" if passed else "fail",
        "output_chars": len(completion),
        "code_chars": len(code),
        "elapsed": time.time() - rollout_time,
    }


def score_batch_outputs(
    batch: list[dict],
    outputs: list[Any],
    timeout: int,
    test_workers: int,
    *,
    rollout_progress: bool = False,
) -> list[dict[str, Any]]:
    completions_by_rec = [[item.text for item in output.outputs] for output in outputs]
    scores: list[dict[str, Any]] = [
        {
            "completions": completions,
            "pass_count": 0,
            "passed_indices": [],
            "test_elapsed": 0.0,
        }
        for completions in completions_by_rec
    ]

    if test_workers <= 1:
        for rec_idx, (rec, completions) in enumerate(zip(batch, completions_by_rec)):
            if rollout_progress:
                log(
                    f"  testing problem source={str(rec.get('source')):<8} "
                    f"uid={record_uid(rec)}"
                )
            score_time = time.time()
            pass_count, passed_indices = score_completions(
                completions,
                rec.get("tests", []),
                timeout=timeout,
                rollout_progress=rollout_progress,
                record_label=record_uid(rec),
            )
            scores[rec_idx]["pass_count"] = pass_count
            scores[rec_idx]["passed_indices"] = passed_indices
            scores[rec_idx]["test_elapsed"] = time.time() - score_time
        return scores

    futures = {}
    with ThreadPoolExecutor(max_workers=test_workers) as executor:
        for rec_idx, (rec, completions) in enumerate(zip(batch, completions_by_rec)):
            if rollout_progress:
                log(
                    f"  testing problem source={str(rec.get('source')):<8} "
                    f"uid={record_uid(rec)}"
                )
            tests = rec.get("tests", [])
            for rollout_idx, completion in enumerate(completions):
                future = executor.submit(
                    score_rollout_job,
                    rec_idx,
                    rollout_idx,
                    completion,
                    tests,
                    timeout,
                )
                futures[future] = (rec_idx, rollout_idx)

        for future in as_completed(futures):
            rec_idx, rollout_idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "rec_idx": rec_idx,
                    "rollout_idx": rollout_idx,
                    "passed": False,
                    "status": f"error:{type(exc).__name__}",
                    "output_chars": len(completions_by_rec[rec_idx][rollout_idx]),
                    "code_chars": 0,
                    "elapsed": 0.0,
                }

            rec_idx = int(result["rec_idx"])
            rollout_idx = int(result["rollout_idx"])
            if result["passed"]:
                scores[rec_idx]["passed_indices"].append(rollout_idx)
            scores[rec_idx]["test_elapsed"] = max(
                float(scores[rec_idx]["test_elapsed"]),
                float(result["elapsed"]),
            )

            if rollout_progress:
                label = record_uid(batch[rec_idx])
                log(
                    f"    rollout {rollout_idx + 1:>2}/{len(completions_by_rec[rec_idx]):<2}"
                    f" {label} status={str(result['status']):<7} "
                    f"output_chars={int(result['output_chars']):>5} "
                    f"code_chars={int(result['code_chars']):>5} "
                    f"time={float(result['elapsed']):.2f}s"
                )

    for score in scores:
        score["passed_indices"].sort()
        score["pass_count"] = len(score["passed_indices"])
    return scores


def scored_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    uids: set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            uid = record_uid(rec)
            if uid:
                uids.add(uid)
    return uids


def scored_eligible_uids(path: Path, pass_counts: set[int], n_samples: int) -> set[str]:
    if not path.exists():
        return set()
    uids: set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                rec_pass_count = int(rec.get("pass_count", -1))
                rec_n_samples = int(rec.get("n_samples", -1))
            except Exception:
                continue
            if rec_n_samples != n_samples or rec_pass_count not in pass_counts:
                continue
            uid = record_uid(rec)
            if uid:
                uids.add(uid)
    return uids


def thinking_template_value(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def format_prompt(
    tokenizer: Any,
    rec: dict,
    enable_thinking: bool | None,
    system_prompt_mode: str,
) -> str:
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages_with_test_contract(rec, system_prompt_mode=system_prompt_mode),
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def score_pool(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.python_log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d] %(message)s",
        force=True,
    )

    pool_path = Path(args.pool)
    scored_path = Path(args.scored)
    scored_path.parent.mkdir(parents=True, exist_ok=True)

    log_step(f"Loading pool from {pool_path}")
    pool = load_jsonl(pool_path)
    log_step(f"Pool loaded: total={len(pool):,}")

    done = scored_uids(scored_path)
    eligible_pass_counts = parse_pass_counts(args.eligible_pass_counts)
    eligible_count = len(
        scored_eligible_uids(
            scored_path,
            pass_counts=eligible_pass_counts,
            n_samples=args.n_samples,
        )
    )
    remaining = [rec for rec in pool if record_uid(rec) not in done]
    if args.limit is not None:
        remaining = remaining[: args.limit]
    log_step(f"Already scored: {len(done):,}")
    log_step(
        "Already eligible: "
        f"{eligible_count:,} "
        f"(pass_count in {sorted(eligible_pass_counts)}, n_samples={args.n_samples})"
    )
    if args.stop_after_eligible is not None:
        log_step(f"Eligible stop target: {args.stop_after_eligible:,}")
        if eligible_count >= args.stop_after_eligible:
            log_step("Eligible stop target already reached; nothing to score.")
            return
    log_step(f"Remaining this run: {len(remaining):,}")

    if not remaining:
        log_step("Nothing to score.")
        return

    log_step("Importing transformers.AutoTokenizer")
    import_start = time.time()
    from transformers import AutoTokenizer
    log_step(f"Imported transformers.AutoTokenizer in {time.time() - import_start:.1f}s")

    log_step("Importing vllm.LLM and vllm.SamplingParams")
    import_start = time.time()
    from vllm import LLM, SamplingParams
    log_step(f"Imported vLLM in {time.time() - import_start:.1f}s")

    enable_thinking = thinking_template_value(args.enable_thinking)
    if enable_thinking is None:
        log_step("Chat template: default enable_thinking behavior")
    else:
        log_step(f"Chat template: enable_thinking={enable_thinking}")

    log_step(f"Loading tokenizer from {args.model}")
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    log_step(f"Tokenizer loaded in {time.time() - tokenizer_start:.1f}s")

    log_step(
        "Initializing vLLM engine "
        f"model={args.model} dtype=bfloat16 "
        f"max_model_len={args.max_model_len} "
        f"gpu_memory_utilization={args.gpu_memory_utilization}"
    )
    llm_start = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    log_step(f"vLLM engine ready in {time.time() - llm_start:.1f}s")
    log_step("Creating vLLM sampling params")
    params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    log("\n=== Rollout settings ===")
    log(f"  n_samples:        {args.n_samples}")
    log(f"  temperature:      {args.temperature}")
    log(f"  top_p/top_k:      {args.top_p} / {args.top_k}")
    log(f"  max_tokens:       {args.max_tokens}")
    log(f"  max_model_len:    {args.max_model_len}")
    log(f"  batch_size:       {args.batch_size}")
    log(f"  test_workers:     {args.test_workers}")
    log(f"  system_prompt:    {args.system_prompt_mode}")
    log(f"  progress_every:   {args.progress_every}")
    log(f"  rollout_progress: {args.rollout_progress}")
    log(
        "  eligible_stop:    "
        + (
            "off"
            if args.stop_after_eligible is None
            else (
                f"{eligible_count:,}/{args.stop_after_eligible:,} "
                f"pass_counts={sorted(eligible_pass_counts)}"
            )
        )
    )

    start_time = time.time()
    scored_this_run = 0
    batch_total = (len(remaining) + args.batch_size - 1) // args.batch_size

    with scored_path.open("a") as out_f:
        for batch_start in range(0, len(remaining), args.batch_size):
            if (
                args.stop_after_eligible is not None
                and eligible_count >= args.stop_after_eligible
            ):
                log_step(
                    f"Eligible stop target reached before next batch: "
                    f"{eligible_count:,}/{args.stop_after_eligible:,}"
                )
                break

            batch_time = time.time()
            batch = remaining[batch_start : batch_start + args.batch_size]
            batch_idx = batch_start // args.batch_size + 1
            if args.rollout_progress or args.batch_progress:
                log_step(
                    f"Preparing batch {batch_idx}/{batch_total}: "
                    f"problems={len(batch)} rollouts_per_problem={args.n_samples}"
                )
            if args.rollout_progress:
                for idx, rec in enumerate(batch, start=1):
                    preview = question_preview(rec, args.question_preview_chars)
                    preview_suffix = f" q='{preview}'" if preview else ""
                    log(
                        f"    queued problem {idx:>3}/{len(batch):<3} "
                        f"source={str(rec.get('source')):<8} "
                        f"uid={record_uid(rec)} "
                        f"tests={len(rec.get('tests', [])):>2}"
                        f"{preview_suffix}"
                    )
            format_start = time.time()
            prompts = [
                format_prompt(
                    tokenizer,
                    rec,
                    enable_thinking=enable_thinking,
                    system_prompt_mode=args.system_prompt_mode,
                )
                for rec in batch
            ]
            if args.rollout_progress or args.batch_progress:
                prompt_tokens = []
                if args.print_prompt_tokens:
                    for prompt in prompts:
                        prompt_tokens.append(len(tokenizer.encode(prompt)))
                token_suffix = (
                    f" prompt_tokens=min/avg/max "
                    f"{min(prompt_tokens)}/{sum(prompt_tokens) // len(prompt_tokens)}/{max(prompt_tokens)}"
                    if prompt_tokens
                    else ""
                )
                log_step(
                    f"Formatted batch {batch_idx}/{batch_total} "
                    f"in {time.time() - format_start:.2f}s{token_suffix}"
                )
                log_step(f"Submitting batch {batch_idx}/{batch_total} to vLLM.generate")
            outputs = llm.generate(prompts, params)
            generate_elapsed = time.time() - batch_time
            if args.rollout_progress or args.batch_progress:
                log_step(
                    f"vLLM.generate returned for batch {batch_idx}/{batch_total} "
                    f"in {generate_elapsed:.1f}s; testing rollouts"
                )

            test_start = time.time()
            if args.batch_progress and args.test_workers > 1:
                test_jobs = sum(len(output.outputs) for output in outputs)
                log_step(
                    f"Testing batch {batch_idx}/{batch_total}: "
                    f"jobs={test_jobs} workers={args.test_workers}"
                )
            batch_scores = score_batch_outputs(
                batch,
                outputs,
                timeout=args.test_timeout,
                test_workers=args.test_workers,
                rollout_progress=args.rollout_progress,
            )
            test_elapsed = time.time() - test_start
            if args.batch_progress:
                log_step(
                    f"Finished testing batch {batch_idx}/{batch_total} "
                    f"in {test_elapsed:.1f}s"
                )

            for rec, score in zip(batch, batch_scores):
                messages = messages_with_test_contract(
                    rec,
                    system_prompt_mode=args.system_prompt_mode,
                )
                required_symbols = infer_required_symbols(
                    [str(test) for test in rec.get("tests", [])]
                )
                completions = score["completions"]
                pass_count = int(score["pass_count"])
                passed_indices = score["passed_indices"]
                result = {
                    "uid": record_uid(rec),
                    "question_id": rec.get("question_id"),
                    "source": rec.get("source"),
                    "messages": messages,
                    "tests": rec.get("tests"),
                    "required_symbols": required_symbols,
                    "pass_count": pass_count,
                    "n_samples": args.n_samples,
                    "pass_rate": pass_count / args.n_samples,
                    "passed_indices": passed_indices,
                    "rollout_model": args.model,
                    "sampling": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_tokens": args.max_tokens,
                        "enable_thinking": args.enable_thinking,
                        "system_prompt_mode": args.system_prompt_mode,
                    },
                }
                if args.save_completions:
                    result["completions"] = completions
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                scored_this_run += 1
                if pass_count in eligible_pass_counts:
                    eligible_count += 1

                if args.progress_every > 0 and scored_this_run % args.progress_every == 0:
                    elapsed = time.time() - start_time
                    total_done = len(done) + scored_this_run
                    speed = scored_this_run / elapsed if elapsed > 0 else 0.0
                    remaining_n = len(remaining) - scored_this_run
                    eta = remaining_n / speed if speed > 0 else 0.0
                    problem_test_elapsed = float(score["test_elapsed"])
                    preview = question_preview(rec, args.question_preview_chars)
                    preview_suffix = f" q='{preview}'" if preview else ""
                    eligible_suffix = (
                        ""
                        if args.stop_after_eligible is None
                        else f" eligible={eligible_count:,}/{args.stop_after_eligible:,}"
                    )
                    log(
                        f"  problem {scored_this_run:>6}/{len(remaining):<6} "
                        f"global={total_done:>6}/{len(pool):<6} "
                        f"source={str(rec.get('source')):<8} "
                        f"uid={record_uid(rec)} "
                        f"tests={len(rec.get('tests', [])):>2} "
                        f"pass={pass_count:>2}/{args.n_samples:<2} "
                        f"rate={pass_count / args.n_samples:.3f} "
                        f"passed=[{format_indices(passed_indices)}] "
                        f"test_time={problem_test_elapsed:.2f}s "
                        f"speed={speed:.2f} prob/s "
                        f"eta={format_duration(eta)}"
                        f"{eligible_suffix}{preview_suffix}"
                    )

            out_f.flush()
            elapsed = time.time() - start_time
            total_done = len(done) + scored_this_run
            speed = scored_this_run / elapsed if elapsed > 0 else 0.0
            remaining_n = len(remaining) - scored_this_run
            eta = remaining_n / speed if speed > 0 else 0.0
            log(
                f"  [{total_done:>6}/{len(pool):>6}] "
                f"this_run={scored_this_run:>6} "
                f"batch={len(batch):>4} "
                f"gen_time={generate_elapsed:.1f}s "
                f"test_time={test_elapsed:.1f}s "
                f"speed={speed:.2f} prob/s eta={format_duration(eta)}"
                + (
                    ""
                    if args.stop_after_eligible is None
                    else f" eligible={eligible_count:,}/{args.stop_after_eligible:,}"
                )
            )

            if (
                args.stop_after_eligible is not None
                and eligible_count >= args.stop_after_eligible
            ):
                log_step(
                    f"Eligible stop target reached after batch {batch_idx}: "
                    f"{eligible_count:,}/{args.stop_after_eligible:,}. "
                    "Stopping before submitting another batch."
                )
                break

    log(f"\nScored {scored_this_run:,} problems -> {scored_path}")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def print_distribution(rows: list[dict]) -> None:
    by_source = Counter(str(row.get("source")) for row in rows)
    by_source_pc = Counter(
        (str(row.get("source")), int(row.get("pass_count", -1))) for row in rows
    )

    log("\n=== Scored distribution ===")
    for source, count in sorted(by_source.items()):
        log(f"  {source}: {count:,}")
        max_pc = max(
            (pc for (src, pc), n in by_source_pc.items() if src == source and n),
            default=0,
        )
        for pc in range(0, max_pc + 1):
            n = by_source_pc.get((source, pc), 0)
            if n:
                log(f"    pass_count={pc:>2}: {n:,}")


def select_source_equal_buckets(
    rows: list[dict],
    source: str,
    target: int,
    pass_counts: list[int],
    n_samples: int,
    rng: random.Random,
) -> list[dict]:
    eligible = [
        row
        for row in rows
        if row.get("source") == source and int(row.get("pass_count", -1)) in pass_counts
        and int(row.get("n_samples", -1)) == n_samples
    ]
    by_pc = {pc: [] for pc in pass_counts}
    for row in eligible:
        by_pc[int(row["pass_count"])].append(row)

    for bucket in by_pc.values():
        rng.shuffle(bucket)

    base = target // len(pass_counts)
    extra = target % len(pass_counts)
    quotas = {
        pc: base + (1 if idx < extra else 0)
        for idx, pc in enumerate(pass_counts)
    }

    selected: list[dict] = []
    leftovers: dict[int, list[dict]] = {}
    deficit = 0

    log(f"\n[{source}] target={target:,} n_samples={n_samples}")
    for pc in pass_counts:
        bucket = by_pc[pc]
        quota = quotas[pc]
        take = min(quota, len(bucket))
        selected.extend(bucket[:take])
        leftovers[pc] = bucket[take:]
        deficit += quota - take
        log(f"  pass_count={pc}: have={len(bucket):,} quota={quota:,} take={take:,}")

    if deficit > 0:
        log(f"  Refill deficit={deficit:,} from remaining hard buckets")
        refill_pool: list[dict] = []
        for pc in pass_counts:
            refill_pool.extend(leftovers[pc])
        take = min(deficit, len(refill_pool))
        selected.extend(refill_pool[:take])
        deficit -= take

    if deficit > 0:
        log(f"  WARNING: source={source} still short by {deficit:,}")

    return selected[:target]


def select_all_eligible(
    rows: list[dict],
    pass_counts: list[int],
    n_samples: int,
    rng: random.Random,
) -> list[dict]:
    selected = [
        row
        for row in rows
        if int(row.get("n_samples", -1)) == n_samples
        and int(row.get("pass_count", -1)) in pass_counts
    ]
    rng.shuffle(selected)

    log(f"\n[all eligible] n_samples={n_samples} pass_counts={pass_counts}")
    log(f"  selected={len(selected):,}")
    dist = Counter((row.get("source", "unknown"), int(row["pass_count"])) for row in selected)
    for source in sorted({src for src, _pc in dist}):
        source_total = sum(count for (src, _pc), count in dist.items() if src == source)
        log(f"  {source}: {source_total:,}")
        for pc in pass_counts:
            log(f"    pass_count={pc}: {dist.get((source, pc), 0):,}")

    return selected


def tier_and_sample(args: argparse.Namespace) -> None:
    scored_path = Path(args.scored)
    train_path = Path(args.train)
    rng = random.Random(args.seed)

    log(f"\n=== Loading scored rows from {scored_path} ===")
    rows = load_jsonl(scored_path)
    log(f"  Total scored rows: {len(rows):,}")

    if not rows:
        raise SystemExit("No scored rows found.")

    print_distribution(rows)

    if args.apply_source_quota:
        selected: list[dict] = []
        selected.extend(
            select_source_equal_buckets(
                rows=rows,
                source="kodcode",
                target=args.kodcode_target,
                pass_counts=PASS_COUNTS,
                n_samples=args.n_samples,
                rng=rng,
            )
        )
        selected.extend(
            select_source_equal_buckets(
                rows=rows,
                source="opencode",
                target=args.opencode_target,
                pass_counts=PASS_COUNTS,
                n_samples=args.n_samples,
                rng=rng,
            )
        )
    else:
        selected = select_all_eligible(
            rows=rows,
            pass_counts=PASS_COUNTS,
            n_samples=args.n_samples,
            rng=rng,
        )

    output_rows = []
    for row in selected:
        output_rows.append(
            {
                "uid": record_uid(row),
                "messages": training_messages(row),
                "tests": row["tests"],
                "source": row["source"],
                "question_id": row["question_id"],
                "required_symbols": row.get("required_symbols")
                or row.get("required_function_names")
                or infer_required_symbols(row["tests"]),
                "pass_count": int(row["pass_count"]),
                "n_samples": int(row["n_samples"]),
                "pass_rate": row["pass_rate"],
                "tier": f"pass_count_{int(row['pass_count'])}",
            }
        )

    rng.shuffle(output_rows)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log(f"\n=== Training set saved -> {train_path} ===")
    log(f"  Total: {len(output_rows):,}")
    final_dist = Counter((row["source"], row["pass_count"]) for row in output_rows)
    for source in ["kodcode", "opencode"]:
        source_total = sum(
            count for (src, _pc), count in final_dist.items() if src == source
        )
        log(f"  {source}: {source_total:,}")
        for pc in PASS_COUNTS:
            log(f"    pass_count={pc}: {final_dist.get((source, pc), 0):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score and tier Qwen3-8B GRPO data."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--pool", default=str(DEFAULT_POOL_PATH))
    parser.add_argument("--scored", default=str(DEFAULT_SCORED_PATH))
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))

    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--tier-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit.")

    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--enable-thinking",
        choices=["true", "false", "default"],
        default="false",
        help="Qwen3 chat template setting. Default false matches the 1.7B scorer.",
    )
    parser.add_argument(
        "--system-prompt-mode",
        choices=["simple", "train", "preserve"],
        default="simple",
        help=(
            "System prompt used during scoring: simple uses the original scorer prompt, "
            "train uses the strict GRPO final-answer prompt, and preserve keeps the "
            "input row's system prompt."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--test-timeout", type=int, default=DEFAULT_TEST_TIMEOUT)
    parser.add_argument(
        "--test-workers",
        type=int,
        default=DEFAULT_TEST_WORKERS,
        help="Parallel Python test workers. Default 1 preserves serial testing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print one detailed progress line every N scored problems. Use 0 to disable.",
    )
    parser.add_argument(
        "--question-preview-chars",
        type=int,
        default=DEFAULT_QUESTION_PREVIEW_CHARS,
        help="Number of user-prompt characters to include in each progress line. Use 0 to hide.",
    )
    parser.add_argument(
        "--rollout-progress",
        action="store_true",
        help="Print pass/fail details for each rollout during test execution.",
    )
    parser.add_argument(
        "--batch-progress",
        action="store_true",
        help="Print timestamped batch preparation/submission/return status.",
    )
    parser.add_argument(
        "--print-prompt-tokens",
        action="store_true",
        help="With --batch-progress, also print min/avg/max prompt token counts.",
    )
    parser.add_argument(
        "--python-log-level",
        default="INFO",
        help="Python logging level used before importing vLLM, e.g. INFO or DEBUG.",
    )
    parser.add_argument(
        "--save-completions",
        action="store_true",
        help="Save raw rollouts. This makes the scored JSONL very large.",
    )
    parser.add_argument(
        "--eligible-pass-counts",
        default="1,2",
        help=(
            "Comma-separated pass_count values treated as eligible for early stop "
            "and tiering-oriented progress. Default: 1,2."
        ),
    )
    parser.add_argument(
        "--stop-after-eligible",
        type=int,
        default=None,
        help=(
            "Stop scoring after this many unique rows in --scored have n_samples "
            "matching --n-samples and pass_count in --eligible-pass-counts. "
            "Existing scored rows count toward the target. The current batch is "
            "finished before stopping."
        ),
    )

    parser.add_argument(
        "--apply-source-quota",
        action="store_true",
        help="Apply KodCode/OpenCode source quotas during tiering. By default, all eligible rows are kept.",
    )
    parser.add_argument("--kodcode-target", type=int, default=SOURCE_TARGETS["kodcode"])
    parser.add_argument("--opencode-target", type=int, default=SOURCE_TARGETS["opencode"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tier_only:
        tier_and_sample(args)
        return

    score_pool(args)

    if not args.score_only:
        tier_and_sample(args)


if __name__ == "__main__":
    main()
