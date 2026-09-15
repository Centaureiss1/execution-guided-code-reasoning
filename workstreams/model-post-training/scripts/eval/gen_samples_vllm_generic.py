"""Generate evalplus-compatible samples for local baseline models with vLLM.

This script is the native-only baseline counterpart to `gen_samples.py`.
It is intended for local non-Qwen3 models such as:
    - Qwen2.5-Coder-1.5B-Instruct
    - DeepSeek-R1-Distill-Qwen-1.5B

Outputs JSONL records with:
    {"task_id", "solution", "raw_output"}

`raw_output` always preserves the full assistant response used for downstream
filtering/evaluation, while `solution` is an initial lightweight extraction.
"""

import argparse
import json
import re
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DATASET_LOADERS = {
    "humaneval": get_human_eval_plus,
    "mbpp": get_mbpp_plus,
}

FAMILY_CHOICES = ["qwen25_coder", "deepseek_r1_distill"]
MODE_NAME = "native"
QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

FAMILY_DEFAULTS = {
    "qwen25_coder": {
        "max_model_len": 8192,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "max_new_tokens": 2048,
    },
    "deepseek_r1_distill": {
        "max_model_len": 36864,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_new_tokens": 32768,
    },
}


def _get_target_func(prompt: str) -> str | None:
    m = re.search(r"def\s+(\w+)\s*\(", prompt)
    if m:
        return m.group(1)

    non_targets = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "math",
        "max",
        "min",
        "not",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
    }
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped.startswith("assert "):
            continue
        for name in re.findall(r"([A-Za-z_]\w*)\s*\(", stripped):
            if name not in non_targets:
                return name
    return None


def _has_target_def(code: str, func_name: str | None) -> bool:
    if func_name:
        return bool(re.search(rf"def\s+{re.escape(func_name)}\s*\(", code))
    return bool(re.search(r"def\s+\w+\s*\(", code))


def _pick_best_block(blocks: list[str], func_name: str | None) -> str | None:
    for block in reversed(blocks):
        candidate = block.strip()
        if _has_target_def(candidate, func_name):
            return candidate
    for block in reversed(blocks):
        candidate = block.strip()
        if re.search(r"def\s+\w+\s*\(", candidate):
            return candidate
    return blocks[-1].strip() if blocks else None


def _extract_code_blocks(text: str) -> list[str]:
    blocks = re.findall(r"```python\s*\n?(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        blocks = re.findall(r"```\s*\n?(.*?)(?:```|$)", text, re.DOTALL)
    return [block.strip() for block in blocks if block.strip()]


def _looks_like_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True

    code_starts = (
        "@",
        "#",
        "def ",
        "class ",
        "from ",
        "import ",
        "if ",
        "elif ",
        "else:",
        "for ",
        "while ",
        "try:",
        "except ",
        "finally:",
        "with ",
        "return",
        "yield",
        "raise ",
        "assert ",
        "pass",
        "break",
        "continue",
        "global ",
        "nonlocal ",
        "del ",
        '"""',
        "'''",
    )
    if stripped.startswith(code_starts):
        return True
    if stripped in {")", "]", "}", "),", "],", "},", "..."}:
        return True
    if re.match(r"[A-Za-z_][\w,\s\[\]]*\s*(?:=|\+=|-=|\*=|/=|//=|%=|:=)", stripped):
        return True
    if re.match(r"(?:\w+\.)?\w+\s*\(", stripped):
        return True
    return False


def _trim_code_candidate(candidate: str) -> str:
    lines = candidate.replace("```", "").strip().splitlines()
    kept: list[str] = []
    in_triple_quote = False
    triple_delim = ""
    saw_code = False

    for line in lines:
        stripped = line.strip()

        if in_triple_quote:
            kept.append(line.rstrip())
            if triple_delim and line.count(triple_delim) % 2 == 1:
                in_triple_quote = False
                triple_delim = ""
            continue

        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue

        if _looks_like_code_line(line):
            kept.append(line.rstrip())
            saw_code = True
            for delim in ('"""', "'''"):
                if line.count(delim) % 2 == 1:
                    in_triple_quote = True
                    triple_delim = delim
                    break
            continue

        if saw_code:
            break

    return "\n".join(kept).strip()


def _maybe_extend_to_import(text: str, code_start: int) -> int:
    import_starts = []
    for pattern in (
        r"(?m)^[ \t]*from\s+[A-Za-z_][\w.]*\s+import\s+[A-Za-z_*][\w.*, ]*",
        r"(?m)^[ \t]*import\s+[A-Za-z_][\w.]*",
        r"(?m)^[ \t]*@[\w.]+(?:\(.*\))?",
    ):
        for match in re.finditer(pattern, text):
            if 0 <= code_start - match.start() <= 200:
                import_starts.append(match.start())

    if import_starts:
        return min(import_starts)
    return code_start


def _find_code_starts(text: str, func_name: str | None) -> list[int]:
    patterns = []
    if func_name:
        patterns.append(rf"(?m)^[ \t]*def\s+{re.escape(func_name)}\s*\(")
    patterns.extend(
        [
            r"(?m)^[ \t]*def\s+[A-Za-z_]\w*\s*\(",
            r"(?m)^[ \t]*class\s+[A-Za-z_]\w*\s*(?:\(|:)",
        ]
    )

    for pattern in patterns:
        matches = [m.start() for m in re.finditer(pattern, text)]
        if matches:
            return matches
    return []


def _extract_code_tail(text: str, func_name: str | None) -> str | None:
    starts = _find_code_starts(text, func_name)
    if not starts:
        return None

    for start in reversed(starts):
        candidate = _trim_code_candidate(text[_maybe_extend_to_import(text, start):])
        if candidate and _has_target_def(candidate, func_name):
            return candidate

    for start in reversed(starts):
        candidate = _trim_code_candidate(text[_maybe_extend_to_import(text, start):])
        if candidate and re.search(r"def\s+\w+\s*\(", candidate):
            return candidate

    return None


def extract_initial_solution(text: str, prompt: str) -> str:
    """Do a lightweight extraction without injecting prompt text or new code."""
    func_name = _get_target_func(prompt)

    candidate_texts: list[str] = []
    if "</think>" in text:
        think_removed = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if think_removed:
            candidate_texts.append(think_removed)
    else:
        tag_removed = re.sub(r"</?think>", "", text).strip()
        if tag_removed:
            candidate_texts.append(tag_removed)

    for candidate_text in candidate_texts:
        blocks = _extract_code_blocks(candidate_text)
        if blocks:
            code = _pick_best_block(blocks, func_name)
            if code:
                return _trim_code_candidate(code)

        code_tail = _extract_code_tail(candidate_text, func_name)
        if code_tail:
            return code_tail

    return ""


def get_native_sampling_args(family: str, max_new_tokens: int | None) -> dict:
    defaults = FAMILY_DEFAULTS[family]
    return {
        "temperature": defaults["temperature"],
        "top_p": defaults["top_p"],
        "top_k": defaults["top_k"],
        "repetition_penalty": defaults["repetition_penalty"],
        "max_tokens": max_new_tokens or defaults["max_new_tokens"],
    }


def resolve_max_model_len(family: str, max_model_len: int | None) -> int:
    return max_model_len or FAMILY_DEFAULTS[family]["max_model_len"]


def build_messages(family: str, prompt: str) -> list[dict[str, str]]:
    user_prompt = f"Complete the following Python function:\n\n{prompt}"
    if family == "qwen25_coder":
        return [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    if family == "deepseek_r1_distill":
        return [
            {"role": "user", "content": user_prompt},
        ]
    raise ValueError(f"Unsupported family: {family}")


def build_prompt_text(tokenizer, family: str, prompt: str) -> tuple[str, str]:
    messages = build_messages(family, prompt)
    if family == "deepseek_r1_distill":
        # Use plain assistant continuation so the model can decide whether to
        # enter a thinking pattern, instead of forcing a `<think>` prefix.
        prompt_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": ""}],
            tokenize=False,
            continue_final_message=True,
        )
        return prompt_text, ""

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text, ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--family", required=True, choices=FAMILY_CHOICES)
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override family-specific default context length",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional task cap for smoke tests")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["task_id"])
                except Exception:
                    pass
        if done:
            print(f"Resuming: {len(done)} samples already done, skipping")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=resolve_max_model_len(args.family, args.max_model_len),
        seed=args.seed,
    )

    params = SamplingParams(**get_native_sampling_args(args.family, args.max_new_tokens))
    problems = DATASET_LOADERS[args.dataset]()

    task_ids = []
    prompts_text = []
    prompt_codes = []
    response_prefixes = []

    for task_id, problem in problems.items():
        if task_id in done:
            continue
        prompt = problem["prompt"]
        prompt_text, response_prefix = build_prompt_text(tokenizer, args.family, prompt)
        task_ids.append(task_id)
        prompts_text.append(prompt_text)
        prompt_codes.append(prompt)
        response_prefixes.append(response_prefix)
        if args.limit is not None and len(task_ids) >= args.limit:
            break

    if not task_ids:
        print("All samples already generated.")
        return

    print(f"Generating {len(task_ids)} samples for {args.dataset} [{MODE_NAME}]...")
    outputs = llm.generate(prompts_text, params)

    with open(output_path, "a") as f:
        for task_id, prompt, response_prefix, out in zip(
            task_ids, prompt_codes, response_prefixes, outputs
        ):
            generated = out.outputs[0].text
            raw_output = response_prefix + generated
            solution = extract_initial_solution(raw_output, prompt)
            record = {
                "task_id": task_id,
                "solution": solution,
                "raw_output": raw_output,
            }
            if response_prefix:
                record["response_prefix"] = response_prefix
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
