"""Build the Qwen3-8B GRPO candidate pool.

This is the 8B version of build_grpo_pool.py. It keeps the old 1.7B script
intact and writes a separate pool for direct Qwen3-8B GRPO:

    data/processed/grpo_pool_qwen3_8b.jsonl

Default composition:
  - OpenCodeInstruct: 40k high-quality Python problems
  - KodCode-Light-RL-10K: all eligible problems after filtering

Each output record has:
  {uid, messages, tests, source, question_id, num_tests, question_chars}

The system prompt matches the Qwen3-8B scoring and GRPO training prompt.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "data" / "raw"
DEFAULT_OUT_PATH = ROOT / "data" / "processed" / "grpo_pool_qwen3_8b.jsonl"
DEFAULT_SFT_PATH = ROOT / "data" / "processed" / "sft_train.jsonl"

DEFAULT_SEED = 42
DEFAULT_OCI_TARGET = 40_000
DEFAULT_MIN_TESTS = 2
DEFAULT_MAX_QUESTION_CHARS = 16_000

SYSTEM = (
    "You are an expert Python programmer.\n"
    "Solve the task using exactly this final-answer format:\n\n"
    "<think>\n"
    "reasoning\n"
    "</think>\n\n"
    "```python\n"
    "complete executable Python code\n"
    "```"
)

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


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_fingerprint(text: str, chars: int = 200) -> str:
    return hashlib.md5(normalize(text)[:chars].encode()).hexdigest()


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def parse_oci_tests(unit_tests_str: str) -> list[str]:
    try:
        tests = json.loads(unit_tests_str)
    except Exception:
        return []
    if not isinstance(tests, list):
        return []
    return [str(t).strip() for t in tests if "assert" in str(t)]


def names_from_target(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names.update(names_from_target(elt))
        return names
    return set()


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


def undefined_test_names(test: str, required_symbols: set[str]) -> set[str]:
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return {"<syntax_error>"}

    local_names = defined_names(tree)
    undefined: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        name = node.id
        if (
            name in local_names
            or name in required_symbols
            or name in BUILTIN_NAMES
            or name in KNOWN_EXTERNAL_NAMES
        ):
            continue
        undefined.add(name)
    return undefined


def self_contained_tests(tests: list[str]) -> list[str]:
    required_symbols = set(infer_required_symbols(tests))
    clean: list[str] = []
    for test in tests:
        if not undefined_test_names(test, required_symbols):
            clean.append(test)
    return clean


def add_symbol_contract(question: str, tests: list[str]) -> str:
    names = infer_required_symbols(tests)
    if not names or "Your solution must define" in question:
        return question

    joined = ", ".join(f"`{name}`" for name in names)
    contract = f"Your solution must define these top-level names: {joined}."
    return f"{question.rstrip()}\n\n{contract}"


def pytest_to_test_blocks(test_str: str) -> list[str]:
    """Convert pytest-style KodCode tests to executable body blocks.

    We keep setup lines inside each test function, not just asserts, because
    many KodCode tests define variables immediately before asserting on them.
    """
    blocks: list[str] = []
    current_block: list[str] = []
    in_func = False

    def flush_block() -> None:
        nonlocal current_block
        if any(line.strip().startswith("assert") for line in current_block):
            blocks.append("\n".join(current_block).strip())
        current_block = []

    for line in test_str.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("from solution import") or stripped.startswith(
            "import solution"
        ):
            continue

        if re.match(r"^def test_\w+\s*\(", stripped):
            flush_block()
            in_func = True
            continue

        if in_func:
            if line and not line[0].isspace():
                in_func = False
                flush_block()
                if re.match(r"^def test_\w+\s*\(", stripped):
                    in_func = True
                continue

            dedented = line[4:] if line.startswith("    ") else line
            if dedented.strip() in {"pass", "..."}:
                continue
            current_block.append(dedented.rstrip())

    flush_block()
    return blocks


def keep_by_length(question: str, max_question_chars: int) -> bool:
    return max_question_chars <= 0 or len(question) <= max_question_chars


def load_sft_fingerprints(sft_path: Path) -> set[str]:
    fps: set[str] = set()
    if not sft_path.exists():
        log(f"  SFT file not found at {sft_path}; skipping SFT dedup")
        return fps

    with sft_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for msg in rec.get("messages", []):
                if msg.get("role") == "user":
                    fps.add(text_fingerprint(stringify(msg.get("content"))))
                    break

    log(f"  SFT dedup: {len(fps):,} fingerprints loaded")
    return fps


def load_eval_fingerprints(cache_dir: Path, sft_path: Path) -> set[str]:
    from datasets import load_dataset

    fps: set[str] = set()

    try:
        ds = load_dataset("openai/openai_humaneval", split="test", cache_dir=str(cache_dir))
        for ex in ds:
            fps.add(text_fingerprint(stringify(ex.get("prompt"))))
        log(f"  HumanEval: {len(ds):,} problems loaded")
    except Exception as exc:
        log(f"  HumanEval load failed: {exc}")

    before = len(fps)
    for split in ["train", "test", "validation", "prompt"]:
        try:
            ds = load_dataset(
                "google-research-datasets/mbpp",
                split=split,
                cache_dir=str(cache_dir),
            )
        except Exception:
            continue
        for ex in ds:
            fps.add(text_fingerprint(stringify(ex.get("text"))))
    log(f"  MBPP: {len(fps) - before:,} fingerprints loaded")

    fps |= load_sft_fingerprints(sft_path)
    log(f"  Total decontamination fingerprints: {len(fps):,}")
    return fps


def dedup_by_fingerprint(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for rec in records:
        fp = rec["_fp"]
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(rec)
    return deduped


def load_opencode(
    cache_dir: Path,
    eval_fps: set[str],
    rng: random.Random,
    target: int,
    min_tests: int,
    max_question_chars: int,
    min_average_test_score: float,
) -> list[dict]:
    from datasets import load_dataset

    log("\n[OpenCodeInstruct] Loading...")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", cache_dir=str(cache_dir))
    log(f"  Raw: {len(ds):,}")

    candidates: list[dict] = []
    skipped = {
        "score": 0,
        "tests": 0,
        "test_sanity": 0,
        "empty": 0,
        "length": 0,
        "decontam": 0,
    }

    for ex in ds:
        try:
            score = float(ex.get("average_test_score"))
        except (TypeError, ValueError):
            skipped["score"] += 1
            continue
        if score < min_average_test_score:
            skipped["score"] += 1
            continue

        question = stringify(ex.get("input"))
        if not question:
            skipped["empty"] += 1
            continue
        if not keep_by_length(question, max_question_chars):
            skipped["length"] += 1
            continue

        tests = parse_oci_tests(stringify(ex.get("unit_tests")))
        if len(tests) < min_tests:
            skipped["tests"] += 1
            continue
        tests = self_contained_tests(tests)
        if len(tests) < min_tests:
            skipped["test_sanity"] += 1
            continue

        fp = text_fingerprint(question)
        if fp in eval_fps:
            skipped["decontam"] += 1
            continue

        qid = stringify(ex.get("id")) or fp
        candidates.append(
            {
                "uid": f"opencode:{qid}",
                "question_id": qid,
                "question": question,
                "tests": tests,
                "source": "opencode",
                "_fp": fp,
                "average_test_score": score,
            }
        )

    candidates = dedup_by_fingerprint(candidates)
    log(f"  Eligible after filters/dedup: {len(candidates):,}")
    log(f"  Skipped: {skipped}")

    if len(candidates) > target:
        candidates = rng.sample(candidates, target)
    log(f"  Selected: {len(candidates):,}")
    return candidates


def load_kodcode(
    cache_dir: Path,
    eval_fps: set[str],
    rng: random.Random,
    target: int | None,
    min_tests: int,
    max_question_chars: int,
) -> list[dict]:
    from datasets import load_dataset

    log("\n[KodCode-Light-RL-10K] Loading...")
    ds = load_dataset("KodCode/KodCode-Light-RL-10K", split="train", cache_dir=str(cache_dir))
    log(f"  Raw: {len(ds):,}")

    records: list[dict] = []
    skipped = {"tests": 0, "test_sanity": 0, "empty": 0, "length": 0, "decontam": 0}

    for ex in ds:
        question = stringify(ex.get("question"))
        if not question:
            skipped["empty"] += 1
            continue
        if not keep_by_length(question, max_question_chars):
            skipped["length"] += 1
            continue

        fp = text_fingerprint(question)
        if fp in eval_fps:
            skipped["decontam"] += 1
            continue

        tests = pytest_to_test_blocks(stringify(ex.get("test")))
        if len(tests) < min_tests:
            skipped["tests"] += 1
            continue
        tests = self_contained_tests(tests)
        if len(tests) < min_tests:
            skipped["test_sanity"] += 1
            continue

        qid = stringify(ex.get("question_id")) or fp
        records.append(
            {
                "uid": f"kodcode:{qid}",
                "question_id": qid,
                "question": question,
                "tests": tests,
                "source": "kodcode",
                "_fp": fp,
                "gpt_pass_percentage": ex.get("gpt_pass_percentage"),
            }
        )

    records = dedup_by_fingerprint(records)
    log(f"  Eligible after filters/dedup: {len(records):,}")
    log(f"  Skipped: {skipped}")

    if target is not None and len(records) > target:
        records = rng.sample(records, target)
        log(f"  Selected: {len(records):,}")
    else:
        log(f"  Selected: {len(records):,} (all eligible)")
    return records


def remove_cross_source_duplicates(oci: list[dict], kodcode: list[dict]) -> list[dict]:
    oci_fps = {rec["_fp"] for rec in oci}
    before = len(kodcode)
    kodcode = [rec for rec in kodcode if rec["_fp"] not in oci_fps]
    log(f"\n[Cross-source dedup] KodCode: {before:,} -> {len(kodcode):,}")
    return oci + kodcode


def to_output_record(rec: dict) -> dict:
    question = add_symbol_contract(rec["question"], rec["tests"])
    return {
        "uid": rec["uid"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "tests": rec["tests"],
        "source": rec["source"],
        "question_id": rec["question_id"],
        "num_tests": len(rec["tests"]),
        "question_chars": len(question),
        "fingerprint": rec["_fp"],
        "required_symbols": infer_required_symbols(rec["tests"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3-8B GRPO candidate pool."
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--sft-path", default=str(DEFAULT_SFT_PATH))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--oci-target", type=int, default=DEFAULT_OCI_TARGET)
    parser.add_argument(
        "--kodcode-target",
        type=int,
        default=None,
        help="Optional cap for KodCode. Omit to keep all eligible KodCode rows.",
    )
    parser.add_argument("--min-tests", type=int, default=DEFAULT_MIN_TESTS)
    parser.add_argument(
        "--max-question-chars",
        type=int,
        default=DEFAULT_MAX_QUESTION_CHARS,
        help="Use 0 to disable the question length filter.",
    )
    parser.add_argument("--min-average-test-score", type=float, default=0.9)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Materialize Hugging Face dataset caches and exit without writing the pool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    sft_path = Path(args.sft_path)
    rng = random.Random(args.seed)

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log("=== Loading decontamination references ===")
    eval_fps = load_eval_fingerprints(cache_dir, sft_path)

    oci = load_opencode(
        cache_dir=cache_dir,
        eval_fps=eval_fps,
        rng=rng,
        target=args.oci_target,
        min_tests=args.min_tests,
        max_question_chars=args.max_question_chars,
        min_average_test_score=args.min_average_test_score,
    )
    kodcode = load_kodcode(
        cache_dir=cache_dir,
        eval_fps=eval_fps,
        rng=rng,
        target=args.kodcode_target,
        min_tests=args.min_tests,
        max_question_chars=args.max_question_chars,
    )

    if args.download_only:
        log("\nDownload-only mode finished after loading dataset caches.")
        return

    records = [to_output_record(rec) for rec in remove_cross_source_duplicates(oci, kodcode)]
    rng.shuffle(records)

    with output_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    oci_n = sum(1 for rec in records if rec["source"] == "opencode")
    kodcode_n = sum(1 for rec in records if rec["source"] == "kodcode")
    log(f"\n=== Pool saved -> {output_path} ===")
    log(f"  OpenCodeInstruct: {oci_n:,}")
    log(f"  KodCode:          {kodcode_n:,}")
    log(f"  Total:            {len(records):,}")


if __name__ == "__main__":
    main()
