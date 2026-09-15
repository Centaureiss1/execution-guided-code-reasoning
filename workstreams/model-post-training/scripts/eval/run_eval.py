"""Run eval pipeline for any model: base, SFT, or GRPO LoRA checkpoint.

Pipeline: [merge LoRA if needed] → gen_samples.py → evalplus.evaluate

Usage:
    # Base model
    python scripts/eval/run_eval.py \\
        --model models/Qwen3-1.7B \\
        --output-dir results/base

    # SFT model
    python scripts/eval/run_eval.py \\
        --model models/Qwen3-1.7B/phase1_sft_merged \\
        --output-dir results/sft

    # GRPO LoRA checkpoint (auto-merges adapter with base)
    python scripts/eval/run_eval.py \\
        --model models/Qwen3-1.7B/phase2_grpo_shuffle_v2/v0-.../checkpoint-600 \\
        --base-model models/Qwen3-1.7B/phase1_sft_merged \\
        --output-dir results/grpo_shuffle_600

    # With thinking mode
    python scripts/eval/run_eval.py \\
        --model models/Qwen3-1.7B/phase1_sft_merged \\
        --output-dir results/sft_thinking \\
        --thinking

    # Skip generation, re-run eval only
    python scripts/eval/run_eval.py \\
        --model models/Qwen3-1.7B/phase1_sft_merged \\
        --output-dir results/sft \\
        --skip-gen
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


DATASETS = ["humaneval", "mbpp"]
SCRIPT_DIR = Path(__file__).resolve().parent


MERGE_LORA_RUNNER = r"""
import runpy
import sys

base_model, adapter_path, output_dir, torch_dtype = sys.argv[1:5]
try:
    import transformers.integrations.tensor_parallel as tensor_parallel
    if not hasattr(tensor_parallel, "EmbeddingParallel"):
        tensor_parallel.EmbeddingParallel = tensor_parallel.ColwiseParallel
        print("[eval] Patched transformers.integrations.tensor_parallel.EmbeddingParallel")
except Exception as exc:
    print(f"[eval] Skipping PEFT tensor-parallel compatibility patch: {exc!r}")

sys.argv = [
    "swift.cli.merge_lora",
    "--model", base_model,
    "--adapters", adapter_path,
    "--output_dir", output_dir,
    "--torch_dtype", torch_dtype,
]
runpy.run_module("swift.cli.merge_lora", run_name="__main__")
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model path (full model or LoRA checkpoint dir)")
    parser.add_argument("--base-model", default=None,
                        help="Base model for LoRA merge (required if model is a LoRA adapter)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save results")
    parser.add_argument("--dataset", default="all", choices=["all", "humaneval", "mbpp"],
                        help="Dataset to run")
    parser.add_argument("--thinking", action="store_true",
                        help="Force thinking on")
    parser.add_argument("--auto", action="store_true",
                        help="Auto thinking (model decides)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override generation temperature passed to gen_samples.py")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Override top_p passed to gen_samples.py")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override top_k passed to gen_samples.py")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional task cap for smoke tests")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--force-merge", action="store_true",
                        help="Remove and rebuild output-dir/merged_model")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip generation, re-run eval on existing samples")
    return parser.parse_args()


def run_cmd(cmd):
    print("$ " + " ".join(cmd))
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print(f"[ERROR] Command exited with code {ret}")
    return ret


def is_lora_adapter(model_path: Path) -> bool:
    return (model_path / "adapter_config.json").exists()


def datasets_to_run(dataset_arg: str) -> list[str]:
    return DATASETS if dataset_arg == "all" else [dataset_arg]


def merge_lora(
    base_model: Path,
    adapter_path: Path,
    output_dir: Path,
    torch_dtype: str,
    force: bool,
) -> Path:
    merged_dir = output_dir / "merged_model"
    marker_path = merged_dir / ".cs639_eval_merge_source.json"
    marker = {
        "base_model": str(base_model),
        "adapter_path": str(adapter_path),
        "torch_dtype": torch_dtype,
    }
    if force and merged_dir.exists():
        print(f"  Removing existing merged model: {merged_dir}")
        shutil.rmtree(merged_dir)

    if merged_dir.exists() and (merged_dir / "config.json").exists():
        if marker_path.exists():
            with marker_path.open() as f:
                old_marker = json.load(f)
            if old_marker == marker:
                print(f"  Merged model already exists at {merged_dir}, skipping merge")
                return merged_dir
            print(
                f"[FATAL] Existing merged model at {merged_dir} was created from a "
                "different source. Use a fresh --output-dir or pass --force-merge."
            )
            sys.exit(1)
        print(
            f"[FATAL] Existing merged model at {merged_dir} has no source marker. "
            "Use a fresh --output-dir or pass --force-merge."
        )
        sys.exit(1)

    print(f"\n=== Merging LoRA adapter ===")
    print(f"  Base:    {base_model}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output:  {merged_dir}")

    cmd = [
        sys.executable,
        "-c",
        MERGE_LORA_RUNNER,
        str(base_model),
        str(adapter_path),
        str(merged_dir),
        torch_dtype,
    ]
    if run_cmd(cmd) != 0:
        print("[FATAL] LoRA merge failed")
        sys.exit(1)

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with marker_path.open("w") as f:
        json.dump(marker, f, indent=2)
        f.write("\n")
    return merged_dir


def main():
    args = parse_args()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "thinking" if args.thinking else ("auto" if args.auto else "no_thinking")
    samples_dir = output_dir / "samples" / suffix
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Merge LoRA if needed
    actual_model = str(model_path)
    if is_lora_adapter(model_path):
        if not args.base_model:
            print("[FATAL] --base-model required for LoRA adapter checkpoints")
            sys.exit(1)
        merged = merge_lora(
            Path(args.base_model).resolve(),
            model_path,
            output_dir,
            args.torch_dtype,
            args.force_merge,
        )
        actual_model = str(merged)

    gen_script = SCRIPT_DIR / "gen_samples.py"

    for dataset in datasets_to_run(args.dataset):
        samples_path = samples_dir / f"{dataset}.jsonl"

        # Step 1: Generate samples
        if not args.skip_gen:
            print(f"\n=== Generating [{dataset}] [{suffix}] ===")
            cmd = [
                sys.executable, str(gen_script),
                "--model", actual_model,
                "--dataset", dataset,
                "--output", str(samples_path),
            ]
            if args.thinking:
                cmd.append("--thinking")
            elif args.auto:
                cmd.append("--auto")
            if args.max_new_tokens:
                cmd += ["--max-new-tokens", str(args.max_new_tokens)]
            if args.temperature is not None:
                cmd += ["--temperature", str(args.temperature)]
            if args.top_p is not None:
                cmd += ["--top-p", str(args.top_p)]
            if args.top_k is not None:
                cmd += ["--top-k", str(args.top_k)]
            cmd += [
                "--max-model-len", str(args.max_model_len),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--tensor-parallel-size", str(args.tensor_parallel_size),
                "--seed", str(args.seed),
            ]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            if run_cmd(cmd) != 0:
                print(f"[ERROR] Generation failed for {dataset}, skipping eval")
                continue

        if not samples_path.exists():
            print(f"[ERROR] No samples at {samples_path}")
            continue

        # Step 2: Evaluate with evalplus (no sanitize — direct eval)
        print(f"\n=== Evaluating [{dataset}] [{suffix}] ===")
        cmd = [
            sys.executable, "-m", "evalplus.evaluate",
            "--dataset", dataset,
            "--samples", str(samples_path),
        ]
        if run_cmd(cmd) != 0:
            print(f"[ERROR] Evaluation failed for {dataset}")


if __name__ == "__main__":
    main()
