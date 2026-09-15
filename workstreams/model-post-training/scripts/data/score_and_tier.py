"""Score grpo_pool.jsonl with the SFT model (vllm), then tier by pass_rate.

流程:
  1. Load grpo_pool.jsonl (~50K records from build_grpo_pool.py)
  2. Format prompts (enable_thinking=False for faster/consistent scoring)
  3. vllm batch inference, n=8 samples per problem, max_tokens=512
  4. Execute completions against tests, compute pass_rate
  5. Save scored results with checkpoint every SAVE_EVERY problems (resumable)
  6. Tier split: Warmup(40-70%) / Main(15-40%) / Push(5-15%)
  7. Save grpo_train.jsonl

Usage:
    python scripts/data/score_and_tier.py            # score + tier
    python scripts/data/score_and_tier.py --score-only
    python scripts/data/score_and_tier.py --tier-only
    python scripts/data/score_and_tier.py --n-samples 4

Output:
    data/processed/grpo_pool_scored.jsonl   # all problems with pass_rate
    data/processed/grpo_train.jsonl          # tiered training set (~20K)
"""

import argparse
import json
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────────────

MODEL_PATH  = "/home/brui/cs639_final/models/Qwen3-1.7B/phase1_sft_merged"
POOL_PATH   = Path("/home/brui/cs639_final/data/processed/grpo_pool.jsonl")
SCORED_PATH = Path("/home/brui/cs639_final/data/processed/grpo_pool_scored.jsonl")
TRAIN_PATH  = Path("/home/brui/cs639_final/data/processed/grpo_train.jsonl")

N_SAMPLES   = 8      # 每题推理次数（pass_rate 分辨率：0/8, 1/8, ..., 8/8）
TEMPERATURE = 0.8
MAX_TOKENS  = 512    # 打分用短补全；GRPO 训练时用 3000
BATCH_SIZE  = 1000   # vllm 一次提交的请求数
SAVE_EVERY  = 500    # 每打分 N 题写一次 checkpoint
SEED        = 42

# 两档边界（对齐 n=8 步长）
# Warmup: 3-6/8 passes（模型能解出，但不总是对）
# Main:   1-2/8 passes（模型偶尔能解出）
# 0/8 和 7-8/8 均跳过（无方差，无学习信号）
WARMUP_LO, WARMUP_HI = 0.375, 0.876   # 包含 6/8=0.750，排除 7/8=0.875
MAIN_LO,   MAIN_HI   = 0.125, 0.375
PUSH_LO,   PUSH_HI   = 0.0,   0.0     # 废弃，不使用

# 两档：全量使用所有 eligible 题目
WARMUP_N = 99_999
MAIN_N   = 99_999
PUSH_N   = 0


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str | None:
    """从模型输出中提取 Python 代码块。先去掉 </think> 前缀。"""
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 退化：找第一个 def
    m2 = re.search(r"(def\s+\w+.*)", text, re.DOTALL)
    if m2:
        return textwrap.dedent(m2.group(0)).strip()
    return None


def run_tests(code: str, tests: list[str], timeout: int = 5) -> bool:
    """拼接 code + assert 语句，用 subprocess 执行，返回是否全部通过。"""
    import tempfile
    full = code + "\n\n" + "\n".join(tests)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            r = subprocess.run(
                [sys.executable, "-c", full],
                timeout=timeout,
                capture_output=True,
                cwd=tmpdir,   # 隔离：代码创建的文件写到临时目录，自动清理
            )
        return r.returncode == 0
    except Exception:
        return False


def score_problem(completions: list[str], tests: list[str]) -> float:
    """计算 n 个 completion 中通过所有 tests 的比例。"""
    passes = sum(
        1 for c in completions
        if (code := extract_code(c)) and run_tests(code, tests)
    )
    return passes / len(completions)


# ── 断点续传 ──────────────────────────────────────────────────────────────────

def load_scored_ids() -> set[str]:
    """读已有的 SCORED_PATH，返回已打分题目的 question_id 集合。"""
    if not SCORED_PATH.exists():
        return set()
    ids = set()
    with open(SCORED_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                ids.add(rec["question_id"])
            except Exception:
                pass
    return ids


# ── 打分主循环 ────────────────────────────────────────────────────────────────

def score_pool(n_samples: int):
    """加载 pool，跳过已打分题，批量推理，checkpoint 写入。"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n=== Loading pool from {POOL_PATH} ===")
    with open(POOL_PATH) as f:
        pool = [json.loads(l) for l in f]
    print(f"  Total pool: {len(pool):,}")

    done_ids = load_scored_ids()
    remaining = [r for r in pool if r["question_id"] not in done_ids]
    print(f"  Already scored: {len(done_ids):,}  |  Remaining: {len(remaining):,}")

    if not remaining:
        print("  All problems already scored. Run --tier-only to build training set.")
        return

    print(f"\n=== Loading model {MODEL_PATH} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )
    params = SamplingParams(n=n_samples, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    def format_prompt(rec: dict) -> str:
        return tokenizer.apply_chat_template(
            rec["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    print(f"\n=== Scoring {len(remaining):,} problems (n={n_samples}) ===")
    est_h = len(remaining) * n_samples * 300 / 4015 / 3600
    print(f"  Estimated time: {est_h:.1f}h @ 4015 tok/s\n")

    scored_this_run = 0
    t_start = time.time()
    out_f = open(SCORED_PATH, "a")

    def _fmt_duration(seconds: float) -> str:
        h, m = divmod(int(seconds), 3600)
        m, s = divmod(m, 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        return f"{m}m {s:02d}s"

    try:
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start : batch_start + BATCH_SIZE]
            prompts = [format_prompt(r) for r in batch]

            outputs = llm.generate(prompts, params)

            for rec, out in zip(batch, outputs):
                completions = [o.text for o in out.outputs]
                pass_rate = score_problem(completions, rec["tests"])
                result = {
                    "question_id": rec["question_id"],
                    "pass_rate":   pass_rate,
                    "source":      rec["source"],
                    "messages":    rec["messages"],
                    "tests":       rec["tests"],
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                scored_this_run += 1

            out_f.flush()
            elapsed = time.time() - t_start
            done_total = len(done_ids) + scored_this_run
            speed = scored_this_run / elapsed if elapsed > 0 else 0
            remaining_n = len(remaining) - scored_this_run
            eta = remaining_n / speed if speed > 0 else 0
            pct = 100 * done_total / len(pool)
            print(f"  [{done_total:>6}/{len(pool):>6}] {pct:5.1f}%  "
                  f"elapsed {_fmt_duration(elapsed)}  "
                  f"speed {speed:.1f} prob/s  "
                  f"ETA {_fmt_duration(eta)}")

    finally:
        out_f.close()

    total_elapsed = time.time() - t_start
    print(f"\n  Scored {scored_this_run:,} problems in {_fmt_duration(total_elapsed)}.")
    print(f"  Results appended → {SCORED_PATH}")


# ── 分档 & 输出 ───────────────────────────────────────────────────────────────

def tier_and_sample():
    """读 SCORED_PATH，按 pass_rate 分三档，随机采样，输出 TRAIN_PATH。"""
    import random
    rng = random.Random(SEED)

    print(f"\n=== Loading scored pool from {SCORED_PATH} ===")
    scored = []
    with open(SCORED_PATH) as f:
        for line in f:
            try:
                scored.append(json.loads(line))
            except Exception:
                pass
    print(f"  Total scored: {len(scored):,}")

    # ── 打印分布 ──────────────────────────────────────────────────────────────
    buckets = {"0-5%": 0, "5-15%": 0, "15-40%": 0, "40-70%": 0, "70-100%": 0}
    for r in scored:
        pr = r["pass_rate"]
        if pr < 0.05:   buckets["0-5%"]    += 1
        elif pr < 0.15: buckets["5-15%"]   += 1
        elif pr < 0.40: buckets["15-40%"]  += 1
        elif pr < 0.70: buckets["40-70%"]  += 1
        else:           buckets["70-100%"] += 1

    print(f"\n=== Pass Rate Distribution ({len(scored):,} problems) ===")
    total = len(scored)
    for k, v in buckets.items():
        bar = "█" * (v * 40 // max(total, 1))
        print(f"  {k:>10}: {v:5d} ({100*v/total:4.1f}%)  {bar}")

    warmup_pool = [r for r in scored if WARMUP_LO <= r["pass_rate"] < WARMUP_HI]
    main_pool   = [r for r in scored if MAIN_LO   <= r["pass_rate"] < MAIN_HI]
    push_pool   = [r for r in scored if PUSH_LO   <= r["pass_rate"] < PUSH_HI]

    print(f"\n  Eligible → Warmup({WARMUP_LO:.0%}-{WARMUP_HI:.0%}): {len(warmup_pool):,}"
          f"  Main({MAIN_LO:.0%}-{MAIN_HI:.0%}): {len(main_pool):,}"
          f"  Push({PUSH_LO:.0%}-{PUSH_HI:.0%}): {len(push_pool):,}")

    # ── 随机采样 ──────────────────────────────────────────────────────────────
    rng.shuffle(warmup_pool)
    rng.shuffle(main_pool)
    rng.shuffle(push_pool)

    warmup_sel = warmup_pool[:WARMUP_N]
    main_sel   = main_pool[:MAIN_N]
    push_sel   = push_pool[:PUSH_N]

    if len(warmup_sel) < WARMUP_N:
        print(f"  WARNING: Warmup only has {len(warmup_sel):,} / {WARMUP_N:,} target. "
              f"Consider relaxing WARMUP boundaries.")
    if len(main_sel) < MAIN_N:
        print(f"  WARNING: Main only has {len(main_sel):,} / {MAIN_N:,} target. "
              f"Consider relaxing MAIN boundaries.")
    if len(push_sel) < PUSH_N:
        print(f"  WARNING: Push only has {len(push_sel):,} / {PUSH_N:,} target. "
              f"Consider relaxing PUSH boundaries or setting PUSH_N=0.")

    # ── 写输出 ────────────────────────────────────────────────────────────────
    records = []
    for tier, items in [("warmup", warmup_sel), ("main", main_sel), ("push", push_sel)]:
        for r in items:
            records.append({
                "messages":    r["messages"],
                "tests":       r["tests"],
                "source":      r["source"],
                "question_id": r["question_id"],
                "pass_rate":   r["pass_rate"],
                "tier":        tier,
            })

    rng.shuffle(records)

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    warmup_n = sum(1 for r in records if r["tier"] == "warmup")
    main_n   = sum(1 for r in records if r["tier"] == "main")
    push_n   = sum(1 for r in records if r["tier"] == "push")

    print(f"\n=== Training set saved → {TRAIN_PATH} ===")
    print(f"  Warmup: {warmup_n:,}  Main: {main_n:,}  Push: {push_n:,}  Total: {len(records):,}")
    print(f"\n  Next step: update grpo_config.yaml dataset path and run GRPO training.")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true",
                        help="Only score, skip tiering")
    parser.add_argument("--tier-only", action="store_true",
                        help="Skip scoring, tier from existing grpo_pool_scored.jsonl")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help=f"Completions per problem (default: {N_SAMPLES})")
    args = parser.parse_args()

    if args.tier_only:
        tier_and_sample()
        return

    score_pool(n_samples=args.n_samples)

    if not args.score_only:
        tier_and_sample()


if __name__ == "__main__":
    main()
