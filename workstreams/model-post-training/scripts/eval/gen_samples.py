"""Generate evalplus-compatible samples using Qwen3 with vllm batch inference.

Keeps the original prompt and extraction logic (proven 67.1% SFT HumanEval),
but uses vllm for faster batch inference.
"""

import argparse
import json
import re
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from evalplus.data import get_human_eval_plus, get_mbpp_plus


DATASET_LOADERS = {
    "humaneval": get_human_eval_plus,
    "mbpp": get_mbpp_plus,
}


def _get_target_func(prompt: str) -> str | None:
    """Extract target function name from prompt (handles both HumanEval and MBPP formats)."""
    # HumanEval: "def func_name(..."
    m = re.search(r'def\s+(\w+)\s*\(', prompt)
    if m:
        return m.group(1)
    # MBPP: "assert func_name(..." or "assert not func_name(..." or "assert math.isclose(func_name(..."
    m = re.search(r'assert\s+(\w+)\s*\(', prompt)
    if m and m.group(1) not in ('math', 'not'):
        return m.group(1)
    m = re.search(r'assert\s+(?:not\s+|math\.\w+\s*\(\s*|\(\s*)(\w+)\s*\(', prompt)
    if m:
        return m.group(1)
    return None


def _has_target_def(code: str, func_name: str | None) -> bool:
    """Check if code contains the target function definition."""
    if func_name:
        return bool(re.search(rf'def\s+{func_name}\s*\(', code))
    return code.strip().startswith("def ")


def _pick_best_block(blocks: list[str], func_name: str | None) -> str | None:
    """From multiple code blocks, pick the last one containing the target function def.
    Models typically output: explanation → snippets → 'Final Implementation' block."""
    # Prefer: last block with target function def
    for block in reversed(blocks):
        if _has_target_def(block.strip(), func_name):
            return block.strip()
    # Fallback: last block that has any "def " (likely the real solution)
    for block in reversed(blocks):
        if re.search(r'def\s+\w+\s*\(', block):
            return block.strip()
    # Last resort: last block
    return blocks[-1].strip() if blocks else None


def extract_solution(text: str, prompt: str, thinking: bool) -> str:
    """Extract code solution from model output."""
    if thinking:
        if "</think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        else:
            text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

    func_name = _get_target_func(prompt)

    # Try ```python blocks first, then bare ``` blocks
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\n?(.*?)```", text, re.DOTALL)

    if blocks:
        code = _pick_best_block(blocks, func_name)
        if code:
            return code if _has_target_def(code, func_name) else prompt + code

    return text.strip() if _has_target_def(text.strip(), func_name) else prompt + text.strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--thinking", action="store_true", help="Force thinking on")
    parser.add_argument("--auto", action="store_true", help="Auto thinking (model decides)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Default: 8192 for thinking/auto, 1024 for no-thinking")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vLLM max_model_len / context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override sampling temperature. Default: 0.2 for MBPP, 0 for HumanEval")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional task cap for smoke tests")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.thinking:
        enable_thinking = True
    elif args.auto:
        enable_thinking = None  # model decides
    else:
        enable_thinking = False
    may_think = enable_thinking is not False
    max_new_tokens = args.max_new_tokens or (8192 if may_think else 1024)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already generated task_ids
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

    print(f"Loading vllm model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        seed=args.seed,
        trust_remote_code=True,
        max_num_seqs=32,
    )
    temperature = args.temperature
    if temperature is None:
        temperature = 0.2 if args.dataset == "mbpp" else 0.0

    if temperature == 0:
        params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    else:
        params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=max_new_tokens,
        )

    problems = DATASET_LOADERS[args.dataset]()

    # Prepare prompts for remaining tasks
    task_ids = []
    prompts_text = []
    prompt_codes = []
    for task_id, problem in problems.items():
        if args.limit is not None and len(task_ids) >= args.limit:
            break
        if task_id in done:
            continue
        prompt = problem["prompt"]
        # Original prompt: simple, no "code block" instruction
        messages = [{"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        task_ids.append(task_id)
        prompts_text.append(text)
        prompt_codes.append(prompt)

    if not task_ids:
        print("All samples already generated.")
        return

    print(f"Generating {len(task_ids)} samples for {args.dataset}...")
    outputs = llm.generate(prompts_text, params)

    with open(output_path, "a") as f:
        for task_id, prompt, out in zip(task_ids, prompt_codes, outputs):
            generated = out.outputs[0].text
            solution = extract_solution(generated, prompt, may_think)
            f.write(json.dumps({"task_id": task_id, "solution": solution, "raw_output": generated}) + "\n")
            f.flush()

    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
