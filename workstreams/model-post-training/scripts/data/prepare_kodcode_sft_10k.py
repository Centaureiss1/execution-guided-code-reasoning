"""Build a configurable cold-start SFT set from KodCode-V1-SFT-R1.

Pipeline:
1. Load KodCode/KodCode-V1-SFT-R1 and preview schema/examples.
2. Tokenize question + response with the Qwen2.5-Coder-1.5B tokenizer.
3. Measure <think>...</think> lengths and save distribution plots.
4. Apply hard quality filters.
5. Sample examples by token-length tier with optional stratification.
6. Save ms-swift-ready JSONL plus a metadata-rich JSONL.
7. Print final stats and a few sanity-check samples.

Default outputs are written under /home/brui/cs639_final/data/processed:
  - <output-name>.jsonl
  - <output-name>_meta.jsonl
  - token_length_distribution.png
  - thinking_length_distribution.png
  - final_token_distribution.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/home/brui/cs639_final")
DEFAULT_CACHE_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
LOCAL_TOKENIZER_PATH = ROOT / "models" / "Qwen2.5-Coder-1.5B-Instruct"

DEFAULT_DATASET = "KodCode/KodCode-V1-SFT-R1"
DEFAULT_SPLIT = "train"
DEFAULT_SEED = 42
DEFAULT_HARD_CEILING = 8192
DEFAULT_MAX_SEQ_LENGTH = 12288
DEFAULT_MIN_THINKING_TOKENS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_OUTPUT_NAME = "kodcode_sft_10k"
DEFAULT_SHORT_MAX_TOKENS = 2048
DEFAULT_MEDIUM_MAX_TOKENS = 4096

TIER_TARGETS = {
    "short": 6000,
    "medium": 3000,
    "hard": 1000,
}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
PYTHON_BLOCK_RE = re.compile(r"```python\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"\w+")

QUESTION_CANDIDATES = [
    "question",
    "prompt",
    "instruction",
    "input",
    "problem",
    "query",
]
RESPONSE_CANDIDATES = [
    "response",
    "answer",
    "output",
    "solution",
    "completion",
]
SOURCE_CANDIDATES = [
    "source",
    "dataset",
    "source_dataset",
    "origin",
]
DIFFICULTY_CANDIDATES = [
    "difficulty",
    "level",
    "difficulty_level",
]
OTHER_STRATIFY_CANDIDATES = [
    "category",
    "task_type",
    "domain",
    "topic",
]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    default_tokenizer = (
        str(LOCAL_TOKENIZER_PATH)
        if LOCAL_TOKENIZER_PATH.exists()
        else "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    )

    parser = argparse.ArgumentParser(
        description="Prepare a KodCode cold-start SFT dataset."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--tokenizer", default=default_tokenizer)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hard-ceiling", type=int, default=DEFAULT_HARD_CEILING)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument(
        "--min-thinking-tokens",
        type=int,
        default=DEFAULT_MIN_THINKING_TOKENS,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--short-max-tokens",
        type=int,
        default=DEFAULT_SHORT_MAX_TOKENS,
        help="Upper bound for the short tier. Uses half-open intervals: short is < short-max-tokens.",
    )
    parser.add_argument(
        "--medium-max-tokens",
        type=int,
        default=DEFAULT_MEDIUM_MAX_TOKENS,
        help=(
            "Upper bound for the medium tier. Uses half-open intervals: "
            "medium is [short-max-tokens, medium-max-tokens)."
        ),
    )
    parser.add_argument("--ngram-n", type=int, default=4)
    parser.add_argument(
        "--ngram-repeat-limit",
        type=int,
        default=3,
        help="Drop samples when the same n-gram repeats consecutively more than this many times.",
    )
    parser.add_argument("--short-target", type=int, default=TIER_TARGETS["short"])
    parser.add_argument("--medium-target", type=int, default=TIER_TARGETS["medium"])
    parser.add_argument("--hard-target", type=int, default=TIER_TARGETS["hard"])
    parser.add_argument(
        "--full-stats",
        action="store_true",
        help="Enable heavier full-dataset stats before filtering.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save histogram PNGs. Disabled by default to reduce memory pressure.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit on how many raw examples to process after loading.",
    )
    args = parser.parse_args()
    if args.short_max_tokens <= 0:
        parser.error("--short-max-tokens must be positive.")
    if args.medium_max_tokens <= args.short_max_tokens:
        parser.error("--medium-max-tokens must be greater than --short-max-tokens.")
    if args.hard_ceiling < args.short_max_tokens:
        parser.error("--hard-ceiling must be >= --short-max-tokens.")
    return args


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(stringify_value(item["text"]))
                elif "content" in item:
                    parts.append(stringify_value(item["content"]))
            else:
                parts.append(stringify_value(item))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "text" in content:
            return stringify_value(content["text"])
        if "content" in content:
            return stringify_value(content["content"])
    return stringify_value(content)


def preview_record(record: dict[str, Any], limit: int = 700) -> str:
    raw = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return raw if len(raw) <= limit else raw[: limit - 3] + "..."


def first_existing_key(column_names: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in column_names:
            return name
    return None


def detect_layout(dataset, column_names: list[str]) -> dict[str, Any]:
    sample = dataset[0]
    layout: dict[str, Any] = {
        "uses_conversations": False,
        "uses_messages": False,
        "question_field": None,
        "response_field": None,
        "source_fields": [key for key in SOURCE_CANDIDATES if key in column_names],
        "difficulty_fields": [
            key for key in DIFFICULTY_CANDIDATES if key in column_names
        ],
        "other_stratify_fields": [
            key for key in OTHER_STRATIFY_CANDIDATES if key in column_names
        ],
    }

    if "conversations" in column_names and isinstance(sample.get("conversations"), list):
        question, response = extract_from_conversations(sample.get("conversations", []))
        if question and response:
            layout["uses_conversations"] = True
            return layout

    if "messages" in column_names and isinstance(sample.get("messages"), list):
        question, response = extract_from_messages(sample.get("messages", []))
        if question and response:
            layout["uses_messages"] = True
            return layout

    question_field = first_existing_key(column_names, QUESTION_CANDIDATES)
    response_field = first_existing_key(column_names, RESPONSE_CANDIDATES)
    if not question_field or not response_field:
        raise ValueError(
            "Could not infer question/response fields from dataset columns "
            f"{column_names}. Please update the candidate lists in the script."
        )

    layout["question_field"] = question_field
    layout["response_field"] = response_field
    return layout


def extract_from_conversations(conversations: list[Any]) -> tuple[str, str]:
    question = ""
    response = ""
    for item in conversations:
        if not isinstance(item, dict):
            continue
        speaker = stringify_value(item.get("from")).lower()
        text = content_to_text(item.get("value"))
        if speaker == "human" and not question and text:
            question = text
        elif speaker == "gpt" and not response and text:
            response = text
        if question and response:
            break
    return question.strip(), response.strip()


def extract_from_messages(messages: list[Any]) -> tuple[str, str]:
    question = ""
    response = ""
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = content_to_text(item.get("content"))
        if role == "user" and not question and text:
            question = text
        elif role == "assistant" and not response and text:
            response = text
        if question and response:
            break
    return question.strip(), response.strip()


def extract_optional_meta(example: dict[str, Any]) -> dict[str, str]:
    meta = {
        "source": "",
        "difficulty": "",
        "dataset_field": "",
        "category": "",
        "task_type": "",
        "domain": "",
        "topic": "",
    }

    for key in SOURCE_CANDIDATES:
        value = stringify_value(example.get(key))
        if value:
            meta["source"] = value
            if key == "dataset":
                meta["dataset_field"] = value
            break

    for key in DIFFICULTY_CANDIDATES:
        value = stringify_value(example.get(key))
        if value:
            meta["difficulty"] = value
            break

    for key in OTHER_STRATIFY_CANDIDATES:
        value = stringify_value(example.get(key))
        if value:
            meta[key] = value

    nested_meta = example.get("metadata") or example.get("meta")
    if isinstance(nested_meta, dict):
        if not meta["source"]:
            for key in SOURCE_CANDIDATES:
                value = stringify_value(nested_meta.get(key))
                if value:
                    meta["source"] = value
                    break
        if not meta["difficulty"]:
            for key in DIFFICULTY_CANDIDATES:
                value = stringify_value(nested_meta.get(key))
                if value:
                    meta["difficulty"] = value
                    break
        for key in OTHER_STRATIFY_CANDIDATES:
            if not meta[key]:
                value = stringify_value(nested_meta.get(key))
                if value:
                    meta[key] = value

    return meta


def extract_text_pair(example: dict[str, Any], layout: dict[str, Any]) -> tuple[str, str]:
    if layout["uses_conversations"]:
        return extract_from_conversations(example.get("conversations", []))
    if layout["uses_messages"]:
        return extract_from_messages(example.get("messages", []))
    question = content_to_text(example.get(layout["question_field"]))
    response = content_to_text(example.get(layout["response_field"]))
    return question.strip(), response.strip()


def batched_token_lengths(
    tokenizer,
    texts: list[str],
    batch_size: int,
) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def assign_token_lengths(
    records: list[dict[str, Any]],
    tokenizer,
    batch_size: int,
    field_name: str,
    text_getter,
    collect_values: bool,
) -> list[int]:
    values: list[int] = []
    batch_records: list[dict[str, Any]] = []
    batch_texts: list[str] = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=False,
        )
        for record, ids in zip(batch_records, encoded["input_ids"]):
            length = len(ids)
            record[field_name] = length
            if collect_values:
                values.append(length)
        batch_records.clear()
        batch_texts.clear()

    for record in records:
        batch_records.append(record)
        batch_texts.append(text_getter(record))
        if len(batch_texts) >= batch_size:
            flush_batch()

    flush_batch()
    return values


def percentile(values: list[int], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    low_value = ordered[lower]
    high_value = ordered[upper]
    return low_value + (high_value - low_value) * (position - lower)


def summarize_lengths(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def log_stats(title: str, stats: dict[str, float]) -> None:
    log(title)
    if not stats["count"]:
        log("  No values available.")
        return
    log(f"  count:  {stats['count']:,}")
    log(f"  mean:   {stats['mean']:.2f}")
    log(f"  median: {stats['median']:.2f}")
    log(f"  p25:    {stats['p25']:.2f}")
    log(f"  p75:    {stats['p75']:.2f}")
    log(f"  p90:    {stats['p90']:.2f}")
    log(f"  p95:    {stats['p95']:.2f}")
    log(f"  max:    {stats['max']:.0f}")


def save_histogram(
    values: list[int],
    output_path: Path,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to save histogram PNGs. "
            "Install it in the active environment first."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    if values:
        ax.hist(values, bins="auto", color=color, edgecolor="black", alpha=0.8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def extract_thinking_text(response: str) -> str:
    matches = THINK_RE.findall(response)
    if not matches:
        return ""
    return "\n\n".join(match.strip() for match in matches if match.strip()).strip()


def has_python_code_block(response: str) -> bool:
    return bool(PYTHON_BLOCK_RE.search(response))


def max_consecutive_ngram_repeat_count(text: str, n: int) -> int:
    tokens = TOKEN_RE.findall(text.lower())
    if len(tokens) < n:
        return 0

    max_run = 1
    last_start = len(tokens) - n
    for start in range(last_start + 1):
        ngram = tokens[start : start + n]
        run = 1
        next_start = start + n
        while next_start <= last_start and tokens[next_start : next_start + n] == ngram:
            run += 1
            next_start += n
        if run > max_run:
            max_run = run
    return max_run


def assign_tier(
    total_tokens: int,
    short_max_tokens: int,
    medium_max_tokens: int,
) -> str:
    if total_tokens < short_max_tokens:
        return "short"
    if total_tokens < medium_max_tokens:
        return "medium"
    return "hard"


def discover_stratify_fields(records: list[dict[str, Any]]) -> list[str]:
    preferred_fields = ["difficulty", "source", "dataset_field", "category", "task_type"]
    available = []
    for field in preferred_fields:
        if any(record.get(field) for record in records):
            available.append(field)
    return available


def stratified_sample(
    records: list[dict[str, Any]],
    target: int,
    rng: random.Random,
    stratify_fields: list[str],
    tier_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {
        "target": target,
        "available": len(records),
        "used_stratify_fields": [],
        "warning": "",
    }
    if len(records) <= target:
        if len(records) < target:
            info["warning"] = (
                f"Tier {tier_name}: only {len(records):,} samples available "
                f"for target {target:,}; taking all."
            )
        sampled = list(records)
        rng.shuffle(sampled)
        return sampled, info

    used_fields = []
    for field in stratify_fields:
        if any(record.get(field) for record in records):
            used_fields.append(field)
    info["used_stratify_fields"] = used_fields

    if not used_fields:
        sampled = rng.sample(records, target)
        return sampled, info

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(field) or "__missing__" for field in used_fields)
        groups[key].append(record)

    if len(groups) == 1:
        sampled = rng.sample(records, target)
        return sampled, info

    for group_records in groups.values():
        rng.shuffle(group_records)

    assigned = {key: 0 for key in groups}
    total = len(records)

    if len(groups) <= target:
        for key, group_records in groups.items():
            if group_records:
                assigned[key] = 1

    remaining = target - sum(assigned.values())
    ideals = {key: len(group_records) / total * target for key, group_records in groups.items()}

    while remaining > 0:
        candidates = [
            key for key, group_records in groups.items()
            if assigned[key] < len(group_records)
        ]
        if not candidates:
            break
        key = max(
            candidates,
            key=lambda item: (
                ideals[item] - assigned[item],
                len(groups[item]) - assigned[item],
                str(item),
            ),
        )
        assigned[key] += 1
        remaining -= 1

    sampled: list[dict[str, Any]] = []
    for key in sorted(groups, key=str):
        sampled.extend(groups[key][: assigned[key]])

    if len(sampled) > target:
        sampled = sampled[:target]
    elif len(sampled) < target:
        leftovers = []
        for key, group_records in groups.items():
            leftovers.extend(group_records[assigned[key] :])
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: target - len(sampled)])

    rng.shuffle(sampled)
    return sampled, info


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "datasets is required to load KodCode/KodCode-V1-SFT-R1. "
            "Install it in the active environment first."
        ) from exc

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required to load the Qwen tokenizer. "
            "Install it in the active environment first."
        ) from exc

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    token_plot_path = output_dir / "token_length_distribution.png"
    thinking_plot_path = output_dir / "thinking_length_distribution.png"
    final_plot_path = output_dir / "final_token_distribution.png"
    swift_jsonl_path = output_dir / f"{args.output_name}.jsonl"
    meta_jsonl_path = output_dir / f"{args.output_name}_meta.jsonl"

    log("=== Step 1: Load Dataset ===")
    log(f"Dataset:        {args.dataset}")
    log(f"Split:          {args.split}")
    log(f"Cache dir:      {cache_dir}")
    log(f"SFT max length: {args.max_seq_length}")
    log(f"Hard ceiling:   {args.hard_ceiling}")
    log(
        "Tier bounds:    "
        f"short < {args.short_max_tokens}, "
        f"medium < {args.medium_max_tokens}, "
        f"hard >= {args.medium_max_tokens}"
    )
    log(f"Output name:    {args.output_name}")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=str(cache_dir),
    )
    if args.limit is not None:
        limit = min(args.limit, len(dataset))
        dataset = dataset.select(range(limit))
        log(f"Debug limit:    {limit:,}")

    if len(dataset) == 0:
        raise ValueError(f"Dataset {args.dataset} split {args.split} is empty.")

    column_names = list(dataset.column_names)
    log(f"Total samples:  {len(dataset):,}")
    log(f"Fields:         {column_names}")
    preview_count = min(3, len(dataset))
    for index in range(preview_count):
        log(f"Preview {index + 1}: {preview_record(dataset[index])}")

    layout = detect_layout(dataset, column_names)
    if layout["uses_conversations"]:
        log("Question/response extraction: conversations[from/value] -> human + gpt")
    elif layout["uses_messages"]:
        log("Question/response extraction: messages[user] + messages[assistant]")
    else:
        log(
            "Question/response extraction: "
            f"{layout['question_field']} + {layout['response_field']}"
        )

    log("\n=== Step 2: Tokenize Full Samples ===")
    log(f"Tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    parsed_records: list[dict[str, Any]] = []
    missing_text_rows = 0
    for index, example in enumerate(dataset):
        question, response = extract_text_pair(example, layout)
        if not question or not response:
            missing_text_rows += 1
            continue
        meta = extract_optional_meta(example)
        parsed_records.append(
            {
                "row_id": index,
                "question": question,
                "response": response,
                **meta,
            }
        )

    if missing_text_rows:
        log(f"Skipped rows with missing question/response: {missing_text_rows:,}")
    log(f"Rows with usable text: {len(parsed_records):,}")

    collect_full_lengths = args.full_stats or args.save_plots
    total_lengths = assign_token_lengths(
        parsed_records,
        tokenizer,
        args.batch_size,
        "total_tokens",
        lambda record: f"{record['question']}\n\n{record['response']}",
        collect_values=collect_full_lengths,
    )

    if args.full_stats:
        total_stats = summarize_lengths(total_lengths)
        log_stats("Full sample token length stats:", total_stats)
    else:
        log("Skipped full pre-filter token stats. Use --full-stats to enable them.")

    if args.save_plots:
        save_histogram(
            total_lengths,
            token_plot_path,
            "KodCode Token Length Distribution",
            "Total tokens (question + response)",
            color="#1f77b4",
        )
        log(f"Saved token length plot to: {token_plot_path}")
    else:
        log("Skipped full token distribution plot. Use --save-plots to enable it.")

    log("\n=== Step 3: Measure <think> Lengths ===")
    collect_thinking_lengths = args.full_stats or args.save_plots
    thinking_lengths: list[int] = []
    thinking_batch_records: list[dict[str, Any]] = []
    thinking_batch_texts: list[str] = []

    def flush_thinking_batch() -> None:
        if not thinking_batch_texts:
            return
        encoded = tokenizer(
            thinking_batch_texts,
            add_special_tokens=False,
            truncation=False,
        )
        for record, ids in zip(thinking_batch_records, encoded["input_ids"]):
            length = len(ids)
            record["thinking_tokens"] = length
            if collect_thinking_lengths:
                thinking_lengths.append(length)
        thinking_batch_records.clear()
        thinking_batch_texts.clear()

    for record in parsed_records:
        thinking_text = extract_thinking_text(record["response"])
        record["has_think"] = bool(thinking_text)
        if thinking_text:
            thinking_batch_records.append(record)
            thinking_batch_texts.append(thinking_text)
            if len(thinking_batch_texts) >= args.batch_size:
                flush_thinking_batch()
        else:
            record["thinking_tokens"] = 0

    flush_thinking_batch()

    think_rows = sum(1 for record in parsed_records if record["has_think"])
    log(f"Rows with <think> tags: {think_rows:,}")
    log(f"Rows without <think> tags: {len(parsed_records) - think_rows:,}")

    if args.full_stats:
        thinking_stats = summarize_lengths(thinking_lengths)
        log_stats("Thinking token length stats:", thinking_stats)
    else:
        log("Skipped full pre-filter thinking stats. Use --full-stats to enable them.")

    if args.save_plots:
        save_histogram(
            thinking_lengths,
            thinking_plot_path,
            "KodCode Thinking Length Distribution",
            "Thinking tokens",
            color="#ff7f0e",
        )
        log(f"Saved thinking length plot to: {thinking_plot_path}")
    else:
        log("Skipped thinking distribution plot. Use --save-plots to enable it.")

    log("\n=== Step 4: Quality Filtering ===")
    log("Rule-removal counts below are counted sequentially in the listed order.")
    filter_counts = Counter()
    loop_rule_candidates = 0
    filtered_records: list[dict[str, Any]] = []
    for record in parsed_records:
        if record["total_tokens"] > args.hard_ceiling:
            filter_counts["total_tokens_over_hard_ceiling"] += 1
            continue
        if not record["has_think"]:
            filter_counts["missing_think_tags"] += 1
            continue
        if record["thinking_tokens"] < args.min_thinking_tokens:
            filter_counts["thinking_tokens_lt_50"] += 1
            continue
        if not has_python_code_block(record["response"]):
            filter_counts["missing_python_code_block"] += 1
            continue
        loop_rule_candidates += 1
        max_repeat_count = max_consecutive_ngram_repeat_count(
            record["response"],
            args.ngram_n,
        )
        if max_repeat_count > args.ngram_repeat_limit:
            filter_counts["consecutive_ngram_loop"] += 1
            continue
        record["tier"] = assign_tier(
            record["total_tokens"],
            args.short_max_tokens,
            args.medium_max_tokens,
        )
        filtered_records.append(record)

    log(f"Filtered dataset size: {len(filtered_records):,}")
    log(
        f"  total_tokens > {args.hard_ceiling}: "
        f"{filter_counts['total_tokens_over_hard_ceiling']:,}"
    )
    log(f"  missing <think> tags:             {filter_counts['missing_think_tags']:,}")
    log(
        f"  thinking tokens < {args.min_thinking_tokens}:        "
        f"{filter_counts['thinking_tokens_lt_50']:,}"
    )
    log(
        f"  missing python code block:       "
        f"{filter_counts['missing_python_code_block']:,}"
    )
    log(
        f"  consecutive repeated {args.ngram_n}-gram > {args.ngram_repeat_limit}: "
        f"{filter_counts['consecutive_ngram_loop']:,}"
    )
    if loop_rule_candidates:
        loop_rule_pct = filter_counts["consecutive_ngram_loop"] / loop_rule_candidates * 100
        log(
            f"    loop rule hit rate among checked samples: {loop_rule_pct:.2f}% "
            f"({filter_counts['consecutive_ngram_loop']:,}/{loop_rule_candidates:,})"
        )
        if loop_rule_pct >= 5:
            log("WARNING: loop-rule hit rate is >= 5%; this may still be too aggressive.")

    log("\n=== Step 5: Tiered Sampling ===")
    tier_targets = {
        "short": args.short_target,
        "medium": args.medium_target,
        "hard": args.hard_target,
    }
    stratify_fields = discover_stratify_fields(filtered_records)
    if stratify_fields:
        log(f"Stratification fields in use when available: {stratify_fields}")
    else:
        log("No difficulty/source-style fields detected; falling back to random sampling.")

    tier_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in filtered_records:
        tier_buckets[record["tier"]].append(record)

    sampled_records: list[dict[str, Any]] = []
    for tier_name in ["short", "medium", "hard"]:
        tier_records = tier_buckets.get(tier_name, [])
        sampled_tier, sample_info = stratified_sample(
            tier_records,
            tier_targets[tier_name],
            rng,
            stratify_fields,
            tier_name,
        )
        if sample_info["warning"]:
            log(f"WARNING: {sample_info['warning']}")
        if sample_info["used_stratify_fields"]:
            log(
                f"Tier {tier_name}: sampled {len(sampled_tier):,} / {len(tier_records):,} "
                f"using {sample_info['used_stratify_fields']}"
            )
        else:
            log(
                f"Tier {tier_name}: sampled {len(sampled_tier):,} / {len(tier_records):,} "
                "without stratification"
            )
        sampled_records.extend(sampled_tier)

    rng.shuffle(sampled_records)
    if len(sampled_records) < sum(tier_targets.values()):
        log(
            f"WARNING: final sampled size is {len(sampled_records):,}, below the "
            f"target {sum(tier_targets.values()):,} because one or more tiers were short."
        )

    log("\n=== Step 6: Save JSONL Outputs ===")
    with swift_jsonl_path.open("w", encoding="utf-8") as swift_handle, meta_jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as meta_handle:
        for record in sampled_records:
            messages = [
                {"role": "user", "content": record["question"]},
                {"role": "assistant", "content": record["response"]},
            ]
            swift_handle.write(
                json.dumps({"messages": messages}, ensure_ascii=False) + "\n"
            )
            meta_handle.write(
                json.dumps(
                    {
                        "messages": messages,
                        "total_tokens": record["total_tokens"],
                        "thinking_tokens": record["thinking_tokens"],
                        "tier": record["tier"],
                        "source": record["source"] or record["dataset_field"] or args.dataset,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    log(f"Saved ms-swift JSONL to: {swift_jsonl_path}")
    log(f"Saved metadata JSONL to: {meta_jsonl_path}")

    log("\n=== Step 7: Final Validation ===")
    final_total_lengths = [record["total_tokens"] for record in sampled_records]
    final_thinking_lengths = [record["thinking_tokens"] for record in sampled_records]
    final_tier_counts = Counter(record["tier"] for record in sampled_records)

    log(f"Total samples: {len(sampled_records):,}")
    for tier_name in ["short", "medium", "hard"]:
        log(f"  {tier_name}: {final_tier_counts.get(tier_name, 0):,}")

    if final_total_lengths:
        log(
            "Token length stats: "
            f"mean={statistics.mean(final_total_lengths):.2f}, "
            f"median={statistics.median(final_total_lengths):.2f}, "
            f"max={max(final_total_lengths):,}"
        )
    else:
        log("Token length stats: no samples")

    if final_thinking_lengths:
        log(
            "Thinking length stats: "
            f"mean={statistics.mean(final_thinking_lengths):.2f}, "
            f"median={statistics.median(final_thinking_lengths):.2f}, "
            f"max={max(final_thinking_lengths):,}"
        )
    else:
        log("Thinking length stats: no samples")

    if args.save_plots:
        save_histogram(
            final_total_lengths,
            final_plot_path,
            "Final Selected Token Length Distribution",
            "Total tokens",
            color="#2ca02c",
        )
        log(f"Saved final token distribution to: {final_plot_path}")
    else:
        log("Skipped final token distribution plot. Use --save-plots to enable it.")

    sanity_count = min(3, len(sampled_records))
    if sanity_count:
        log("\nSanity-check samples:")
        sanity_records = rng.sample(sampled_records, sanity_count)
        for index, record in enumerate(sanity_records, start=1):
            question_preview = textwrap.shorten(
                record["question"].replace("\n", " "),
                width=100,
                placeholder="...",
            )
            thinking_preview_text = extract_thinking_text(record["response"])
            thinking_preview = textwrap.shorten(
                thinking_preview_text.replace("\n", " "),
                width=200,
                placeholder="...",
            )
            log(f"  Sample {index} question: {question_preview}")
            log(f"  Sample {index} thinking: {thinking_preview}")


if __name__ == "__main__":
    main()
