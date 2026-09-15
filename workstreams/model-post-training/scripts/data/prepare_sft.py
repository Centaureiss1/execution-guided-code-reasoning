"""Prepare SFT training data from OpenCodeInstruct + Magicoder-OSS-Instruct.

Output: data/processed/sft_train.jsonl
Format: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

CACHE_DIR = "/home/brui/cs639_final/data/raw"
OUTPUT_PATH = Path("/home/brui/cs639_final/data/processed/sft_train.jsonl")
SEED = 42

OPENCODE_TARGET = 50_000
MAGICODER_TARGET = 20_000
MAX_CHARS = 12_000   # ~3000 tokens, fits in 4096 ctx with chat template overhead
MIN_OUTPUT_CHARS = 50


def log(msg):
    print(msg, flush=True)


def prepare_opencode():
    log("\n=== OpenCodeInstruct ===")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", cache_dir=CACHE_DIR)
    log(f"Loaded: {len(ds):,}")

    # 1. Quality filter: average_test_score >= 0.9 (field is stored as string)
    # Note: tests_execution_status is a JSON list of per-test "pass"/"fail" strings,
    # not a single "passed" value — use average_test_score as the sole quality signal.
    def score_filter(x):
        try:
            return float(x["average_test_score"]) >= 0.9
        except (TypeError, ValueError):
            return False
    ds = ds.filter(score_filter)
    log(f"After score >= 0.9: {len(ds):,}")

    # 3. Non-empty filter
    ds = ds.filter(lambda x: bool(x["input"]) and bool(x["output"]))
    log(f"After non-empty: {len(ds):,}")

    # 4 & 5. Length filter (min output + max combined)
    ds = ds.filter(lambda x:
        len(x["output"]) >= MIN_OUTPUT_CHARS and
        len(x["input"]) + len(x["output"]) <= MAX_CHARS
    )
    log(f"After length filter: {len(ds):,}")

    # 6. Exact dedup on input
    seen = set()
    indices = []
    for i, example in enumerate(ds):
        key = example["input"]
        if key not in seen:
            seen.add(key)
            indices.append(i)
    ds = ds.select(indices)
    log(f"After dedup: {len(ds):,}")

    if len(ds) < OPENCODE_TARGET:
        log(f"WARNING: only {len(ds):,} examples after filtering (target {OPENCODE_TARGET:,}). Consider lowering score threshold to 0.8.")
        samples = ds
    else:
        rng = random.Random(SEED)
        selected = rng.sample(range(len(ds)), OPENCODE_TARGET)
        samples = ds.select(selected)

    log(f"Sampled: {len(samples):,}")

    records = [
        {"messages": [
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]}
        for ex in samples
    ]
    return records


def prepare_magicoder():
    log("\n=== Magicoder-OSS-Instruct-75K ===")
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", cache_dir=CACHE_DIR)
    log(f"Loaded: {len(ds):,}")

    # 1. Language filter: Python only (lang field uses lowercase)
    ds = ds.filter(lambda x: x["lang"] == "python")
    log(f"After lang == Python: {len(ds):,}")

    # 2. Non-empty filter
    ds = ds.filter(lambda x: bool(x["problem"]) and bool(x["solution"]))
    log(f"After non-empty: {len(ds):,}")

    # 3 & 4. Length filter
    ds = ds.filter(lambda x:
        len(x["solution"]) >= MIN_OUTPUT_CHARS and
        len(x["problem"]) + len(x["solution"]) <= MAX_CHARS
    )
    log(f"After length filter: {len(ds):,}")

    # 5. Exact dedup on problem
    seen = set()
    indices = []
    for i, example in enumerate(ds):
        key = example["problem"]
        if key not in seen:
            seen.add(key)
            indices.append(i)
    ds = ds.select(indices)
    log(f"After dedup: {len(ds):,}")

    if len(ds) > MAGICODER_TARGET:
        rng = random.Random(SEED)
        selected = rng.sample(range(len(ds)), MAGICODER_TARGET)
        samples = ds.select(selected)
    else:
        samples = ds

    log(f"Sampled: {len(samples):,}")

    records = [
        {"messages": [
            {"role": "user", "content": ex["problem"]},
            {"role": "assistant", "content": ex["solution"]},
        ]}
        for ex in samples
    ]
    return records


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    opencode = prepare_opencode()
    magicoder = prepare_magicoder()

    combined = opencode + magicoder
    random.Random(SEED).shuffle(combined)

    with open(OUTPUT_PATH, "w") as f:
        for record in combined:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(f"\n=== Done ===")
    log(f"OpenCodeInstruct: {len(opencode):,}")
    log(f"Magicoder:        {len(magicoder):,}")
    log(f"Total:            {len(combined):,}")
    log(f"Saved to:         {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
