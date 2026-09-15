"""Run native-only vLLM evals for local baseline models.

Pipeline:
    gen_samples_vllm_generic.py -> evalplus.evaluate
"""

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = ["humaneval", "mbpp"]
FAMILY_CHOICES = ["qwen25_coder", "deepseek_r1_distill"]
MODE_NAME = "native"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAX_MODEL_LEN = {
    "qwen25_coder": 8192,
    "deepseek_r1_distill": 36864,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--family", required=True, choices=FAMILY_CHOICES)
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "humaneval", "mbpp"],
        help="Dataset to run",
    )
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation")
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


def run_cmd(cmd):
    print("$ " + " ".join(cmd))
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print(f"[ERROR] Command exited with code {ret}")
    return ret


def datasets_to_run(dataset_arg: str) -> list[str]:
    return DATASETS if dataset_arg == "all" else [dataset_arg]


def resolve_max_model_len(family: str, max_model_len: int | None) -> int:
    return max_model_len or DEFAULT_MAX_MODEL_LEN[family]


def evaluate_samples(dataset: str, samples_path: Path) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "evalplus.evaluate",
        dataset,
        "--samples",
        str(samples_path),
    ]
    return run_cmd(cmd) == 0


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples" / MODE_NAME
    samples_dir.mkdir(parents=True, exist_ok=True)

    gen_script = SCRIPT_DIR / "gen_samples_vllm_generic.py"

    for dataset in datasets_to_run(args.dataset):
        samples_path = samples_dir / f"{dataset}.jsonl"

        if not args.skip_gen:
            print(f"\n=== Generating [{dataset}] [{MODE_NAME}] ===")
            cmd = [
                sys.executable,
                str(gen_script),
                "--model",
                args.model,
                "--family",
                args.family,
                "--dataset",
                dataset,
                "--output",
                str(samples_path),
                "--max-model-len",
                str(resolve_max_model_len(args.family, args.max_model_len)),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--seed",
                str(args.seed),
            ]
            if args.max_new_tokens is not None:
                cmd += ["--max-new-tokens", str(args.max_new_tokens)]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]

            if run_cmd(cmd) != 0:
                print(f"[ERROR] Generation failed for {dataset}, skipping eval")
                continue

        if not samples_path.exists():
            print(f"[ERROR] No samples at {samples_path}")
            continue

        print(f"\n=== Evaluating [{dataset}] [{MODE_NAME}] ===")
        if not evaluate_samples(dataset, samples_path):
            print(f"[ERROR] Evaluation failed for {dataset}")


if __name__ == "__main__":
    main()
