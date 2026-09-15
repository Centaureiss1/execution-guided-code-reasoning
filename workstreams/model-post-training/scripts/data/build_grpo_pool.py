"""Step 1: Build GRPO candidate pool.

采样 + 去污染 + 去重，输出统一格式的候选池 JSONL。

Sources:
  OpenCodeInstruct: 30K (score>=0.9, assert格式)
  KodCode-Light-RL-10K: 10K (pytest→exec转换)

去污染: 去除与 HumanEval / MBPP 相似的题
去重: OCI 和 KodCode 之间精确去重

Output: data/processed/grpo_pool.jsonl
  每条: {messages, tests, source, question_id}
"""

import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

CACHE_DIR  = "/home/brui/cs639_final/data/raw"
OUT_PATH   = Path("/home/brui/cs639_final/data/processed/grpo_pool.jsonl")
SFT_PATH   = Path("/home/brui/cs639_final/data/processed/sft_train.jsonl")
SEED       = 42
OCI_N      = 40_000   # 从 OpenCodeInstruct 抽多少
MIN_TESTS  = 2        # 每题至少几个 assert

SYSTEM = (
    "You are an expert Python programmer. "
    "Think step by step, then write a complete Python function "
    "in a ```python``` code block."
)

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """用于去重/去污染的文本规范化：小写、去标点、压缩空白。"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_fingerprint(text: str, chars: int = 200) -> str:
    return hashlib.md5(normalize(text)[:chars].encode()).hexdigest()


def parse_oci_tests(unit_tests_str: str) -> list[str]:
    """OpenCodeInstruct unit_tests 字段是 JSON 字符串 list of assert。"""
    try:
        tests = json.loads(unit_tests_str)
        return [t.strip() for t in tests if t.strip().startswith("assert")]
    except Exception:
        return []


def pytest_to_asserts(test_str: str) -> list[str]:
    """KodCode pytest格式 → bare assert 列表。

    输入样例:
        from solution import func
        def test_case1():
            assert func(1) == 2
        def test_case2():
            assert func(0) == 0
    输出:
        ["assert func(1) == 2", "assert func(0) == 0"]
    """
    asserts = []
    in_func = False
    for line in test_str.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过 import 行
        if stripped.startswith("from solution import") or \
           stripped.startswith("import solution"):
            continue
        # 进入 test 函数
        if re.match(r"^def test_\w+\s*\(", stripped):
            in_func = True
            continue
        if in_func:
            # 非缩进非空行 → 退出当前 test 函数
            if line and not line[0].isspace():
                in_func = False
                if re.match(r"^def test_\w+\s*\(", stripped):
                    in_func = True
                continue
            # 提取 assert 行（去掉4空格缩进）
            dedented = line[4:] if line.startswith("    ") else line
            if dedented.strip().startswith("assert"):
                asserts.append(dedented.strip())
    return asserts


# ── 去污染参考集 ──────────────────────────────────────────────────────────────

def load_sft_fingerprints() -> set[str]:
    """加载 SFT 训练集的题目指纹，避免 GRPO pool 与 SFT 数据重叠。"""
    fps = set()
    if not SFT_PATH.exists():
        print(f"  SFT file not found at {SFT_PATH}, skipping SFT dedup")
        return fps
    with open(SFT_PATH) as f:
        for line in f:
            rec = json.loads(line)
            for msg in rec.get("messages", []):
                if msg["role"] == "user":
                    fps.add(text_fingerprint(msg["content"]))
                    break
    print(f"  SFT dedup: {len(fps)} fingerprints loaded from {SFT_PATH.name}")
    return fps


def load_eval_fingerprints() -> set[str]:
    """加载 HumanEval + MBPP + SFT 的题目指纹，用于去污染和去重。"""
    fps = set()

    # HumanEval
    try:
        ds = load_dataset("openai/openai_humaneval", split="test",
                          cache_dir=CACHE_DIR)
        for ex in ds:
            fps.add(text_fingerprint(ex.get("prompt", "")))
        print(f"  HumanEval: {len(ds)} problems loaded")
    except Exception as e:
        print(f"  HumanEval load failed: {e}")

    # MBPP (full split)
    n_before = len(fps)
    for split in ["train", "test", "validation", "prompt"]:
        try:
            ds = load_dataset("google-research-datasets/mbpp", split=split,
                              cache_dir=CACHE_DIR)
            for ex in ds:
                fps.add(text_fingerprint(ex.get("text", "")))
        except Exception:
            pass
    print(f"  MBPP: {len(fps)-n_before} problems loaded")

    # SFT training data
    sft_fps = load_sft_fingerprints()
    fps |= sft_fps

    print(f"  Total decontamination fingerprints: {len(fps)}")
    return fps


# ── OpenCodeInstruct ──────────────────────────────────────────────────────────

def load_oci(n: int, eval_fps: set, rng: random.Random) -> list[dict]:
    print(f"\n[OCI] Loading (score>=0.9, sample {n:,})...")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train",
                      cache_dir=CACHE_DIR)

    candidates = []
    for ex in ds:
        try:
            score = float(ex["average_test_score"])
        except (TypeError, ValueError):
            continue
        if score < 0.9:
            continue
        tests = parse_oci_tests(ex.get("unit_tests", ""))
        if len(tests) < MIN_TESTS:
            continue
        inp = ex.get("input", "").strip()
        if not inp:
            continue
        # 去污染
        if text_fingerprint(inp) in eval_fps:
            continue
        candidates.append({
            "question_id": ex.get("id", ""),
            "question":    inp,
            "tests":       tests,
            "source":      "opencode",
            "_fp":         text_fingerprint(inp),
        })

    print(f"  Eligible: {len(candidates):,}")
    selected = rng.sample(candidates, min(n, len(candidates)))
    print(f"  Sampled:  {len(selected):,}")
    return selected


# ── KodCode ───────────────────────────────────────────────────────────────────

def load_kodcode(eval_fps: set) -> list[dict]:
    print("\n[KodCode] Loading KodCode-Light-RL-10K...")
    ds = load_dataset("KodCode/KodCode-Light-RL-10K", split="train",
                      cache_dir=CACHE_DIR)

    records = []
    skipped_tests = 0
    skipped_decontam = 0
    for ex in ds:
        q = ex.get("question", "").strip()
        if not q:
            continue
        # 去污染
        if text_fingerprint(q) in eval_fps:
            skipped_decontam += 1
            continue
        tests = pytest_to_asserts(ex.get("test", ""))
        if len(tests) < MIN_TESTS:
            skipped_tests += 1
            continue
        records.append({
            "question_id": ex.get("question_id", ""),
            "question":    q,
            "tests":       tests,
            "source":      "kodcode",
            "gpt_pass_pct": ex.get("gpt_pass_percentage", None),
            "_fp":          text_fingerprint(q),
        })

    print(f"  Loaded: {len(records):,}  "
          f"(skipped: {skipped_decontam} decontam, {skipped_tests} no tests)")
    return records


# ── 跨数据集去重 ──────────────────────────────────────────────────────────────

def dedup(oci: list, kc: list) -> tuple[list, list]:
    oci_fps = {r["_fp"] for r in oci}
    before = len(kc)
    kc_deduped = [r for r in kc if r["_fp"] not in oci_fps]
    print(f"\n[Dedup] KodCode: {before} → {len(kc_deduped)} "
          f"({before-len(kc_deduped)} overlaps removed)")
    return oci, kc_deduped


# ── 拼合 & 输出 ───────────────────────────────────────────────────────────────

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("=== Loading decontamination reference (HumanEval + MBPP) ===")
    eval_fps = load_eval_fingerprints()

    oci = load_oci(OCI_N, eval_fps, rng)
    kc  = load_kodcode(eval_fps)
    oci, kc = dedup(oci, kc)

    pool = []
    for r in oci + kc:
        pool.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": r["question"]},
            ],
            "tests":       r["tests"],
            "source":      r["source"],
            "question_id": r["question_id"],
        })

    rng.shuffle(pool)

    with open(OUT_PATH, "w") as f:
        for rec in pool:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    oci_n = sum(1 for r in pool if r["source"] == "opencode")
    kc_n  = sum(1 for r in pool if r["source"] == "kodcode")
    print(f"\n=== Pool saved → {OUT_PATH} ===")
    print(f"  OpenCodeInstruct: {oci_n:,}")
    print(f"  KodCode:          {kc_n:,}")
    print(f"  Total:            {len(pool):,}")


if __name__ == "__main__":
    main()
