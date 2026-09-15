"""Run a Qwen3-friendly LiveCodeBench code-generation pipeline.

This wrapper keeps LiveCodeBench scoring in the official lcb_runner package,
while adapting the local model side for this project:

1. Merge a LoRA checkpoint if --model points at an adapter.
2. Generate LiveCodeBench custom outputs with Qwen3 chat-template thinking.
3. Optionally call the official custom evaluator on those outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


MERGE_LORA_RUNNER = r"""
import runpy
import sys

base_model, adapter_path, output_dir, torch_dtype = sys.argv[1:5]
try:
    import transformers.integrations.tensor_parallel as tensor_parallel
    if not hasattr(tensor_parallel, "EmbeddingParallel"):
        tensor_parallel.EmbeddingParallel = tensor_parallel.ColwiseParallel
        print("[lcb] Patched transformers.integrations.tensor_parallel.EmbeddingParallel")
except Exception as exc:
    print(f"[lcb] Skipping PEFT tensor-parallel compatibility patch: {exc!r}")

sys.argv = [
    "swift.cli.merge_lora",
    "--model", base_model,
    "--adapters", adapter_path,
    "--output_dir", output_dir,
    "--torch_dtype", torch_dtype,
]
runpy.run_module("swift.cli.merge_lora", run_name="__main__")
"""


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    ret = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None).returncode
    if ret != 0:
        raise SystemExit(f"Command failed with exit code {ret}")


def is_lora_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def merge_lora(
    *,
    base_model: Path,
    adapter_path: Path,
    output_dir: Path,
    torch_dtype: str,
    force: bool,
) -> Path:
    merged_dir = output_dir / "merged_model"
    marker_path = merged_dir / ".cs639_lcb_merge_source.json"
    marker = {
        "base_model": str(base_model),
        "adapter_path": str(adapter_path),
        "torch_dtype": torch_dtype,
    }
    if force and merged_dir.exists():
        log(f"Removing existing merged model: {merged_dir}")
        shutil.rmtree(merged_dir)

    if (merged_dir / "config.json").exists():
        if marker_path.exists():
            with marker_path.open() as f:
                old_marker = json.load(f)
            if old_marker == marker:
                log(f"Using existing merged model: {merged_dir}")
                return merged_dir
            raise SystemExit(
                f"Existing merged model at {merged_dir} was created from a "
                "different source. Use a fresh --output-dir or pass --force-merge."
            )
        raise SystemExit(
            f"Existing merged model at {merged_dir} has no source marker. "
            "Use a fresh --output-dir or pass --force-merge."
        )

    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    log("Merging LoRA adapter for LiveCodeBench evaluation")
    run_cmd(
        [
            sys.executable,
            "-c",
            MERGE_LORA_RUNNER,
            str(base_model),
            str(adapter_path),
            str(merged_dir),
            torch_dtype,
        ]
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with marker_path.open("w") as f:
        json.dump(marker, f, indent=2)
        f.write("\n")
    return merged_dir


def lcb_env(lcb_dir: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if lcb_dir:
        lcb_path = str(Path(lcb_dir).resolve())
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = lcb_path if not current else f"{lcb_path}:{current}"
    return env


def validate_lcb_available(lcb_dir: str | None) -> None:
    env = lcb_env(lcb_dir)
    code = (
        "import importlib.util; "
        "spec = importlib.util.find_spec('lcb_runner'); "
        "assert spec is not None; "
        "locations = list(spec.submodule_search_locations or []); "
        "print(locations[0] if locations else spec.origin)"
    )
    ret = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ret.returncode != 0:
        raise SystemExit(
            "Could not import official LiveCodeBench lcb_runner.\n"
            "Clone it and pass --lcb-dir, for example:\n"
            "  git clone https://github.com/LiveCodeBench/LiveCodeBench.git "
            "/tmp/cs639_grpo/LiveCodeBench\n"
            "  --lcb-dir /tmp/cs639_grpo/LiveCodeBench"
        )
    log(f"Using LiveCodeBench package: {ret.stdout.strip()}")


def default_output_name(args: argparse.Namespace) -> str:
    fast = "full" if args.not_fast else "lite"
    temp = str(args.temperature).replace(".", "p")
    subset = f"_{Path(args.subset_file).stem}" if args.subset_file else ""
    dates = ""
    if args.start_date or args.end_date:
        dates = f"_{args.start_date or 'start'}_{args.end_date or 'end'}"
    return (
        f"codegeneration_{args.release_version}_{fast}{dates}{subset}_"
        f"{args.thinking_mode}_n{args.n}_t{temp}.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Full model or LoRA checkpoint")
    parser.add_argument("--base-model", default=None, help="Base model when --model is a LoRA adapter")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lcb-dir", default=os.environ.get("LCB_DIR"))
    parser.add_argument("--custom-output-file", default=None)
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--not-fast", action="store_true")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--streaming-benchmark", action="store_true")
    parser.add_argument("--thinking-mode", choices=["thinking", "auto", "no_thinking"], default="thinking")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--torch-dtype", default="bfloat16", help="dtype used for LoRA merge")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stop", default="###")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run official LCB custom evaluator")
    parser.add_argument("--num-process-evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--force-merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_lcb_available(args.lcb_dir)

    model_path = Path(args.model).resolve()
    actual_model = model_path
    if is_lora_adapter(model_path):
        if not args.base_model:
            raise SystemExit("--base-model is required when --model is a LoRA adapter")
        actual_model = merge_lora(
            base_model=Path(args.base_model).resolve(),
            adapter_path=model_path,
            output_dir=output_dir,
            torch_dtype=args.torch_dtype,
            force=args.force_merge,
        )
    else:
        log(f"Using full model directly: {actual_model}")

    custom_output = (
        Path(args.custom_output_file).resolve()
        if args.custom_output_file
        else output_dir / "custom_outputs" / default_output_name(args)
    )

    if args.evaluate and (args.debug or args.limit is not None):
        raise SystemExit(
            "Official LiveCodeBench custom_evaluator expects one output for every "
            "problem in the selected benchmark. Use --debug/--limit without "
            "--evaluate for smoke generation, then run full generation with "
            "--evaluate."
        )

    if not args.skip_gen:
        gen_script = SCRIPT_DIR / "lcb_qwen3_generate.py"
        cmd = [
            sys.executable,
            str(gen_script),
            "--model",
            str(actual_model),
            "--output",
            str(custom_output),
            "--release-version",
            args.release_version,
            "--thinking-mode",
            args.thinking_mode,
            "--n",
            str(args.n),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--max-model-len",
            str(args.max_model_len),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--dtype",
            args.dtype,
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
            "--stop",
            args.stop,
        ]
        if args.lcb_dir:
            cmd += ["--lcb-dir", str(Path(args.lcb_dir).resolve())]
        if args.not_fast:
            cmd.append("--not-fast")
        if args.start_date:
            cmd += ["--start-date", args.start_date]
        if args.end_date:
            cmd += ["--end-date", args.end_date]
        if args.subset_file:
            cmd += ["--subset-file", str(Path(args.subset_file).resolve())]
        if args.streaming_benchmark:
            cmd.append("--streaming-benchmark")
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.debug:
            cmd.append("--debug")
        run_cmd(cmd)
    else:
        log(f"Skipping generation; using custom output: {custom_output}")

    if args.evaluate:
        if not custom_output.exists():
            raise SystemExit(f"Missing custom output for evaluation: {custom_output}")
        if args.subset_file:
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "lcb_subset_evaluate.py"),
                "--custom-output-file",
                str(custom_output),
                "--subset-file",
                str(Path(args.subset_file).resolve()),
                "--release-version",
                args.release_version,
                "--num-process-evaluate",
                str(args.num_process_evaluate),
                "--timeout",
                str(args.timeout),
            ]
            if args.lcb_dir:
                cmd += ["--lcb-dir", str(Path(args.lcb_dir).resolve())]
            if args.streaming_benchmark:
                cmd.append("--streaming-benchmark")
        else:
            cmd = [
                sys.executable,
                "-m",
                "lcb_runner.runner.custom_evaluator",
                "--scenario",
                "codegeneration",
                "--release_version",
                args.release_version,
                "--custom_output_file",
                str(custom_output),
                "--num_process_evaluate",
                str(args.num_process_evaluate),
                "--timeout",
                str(args.timeout),
            ]
        if args.not_fast:
            cmd.append("--not-fast" if args.subset_file else "--not_fast")
        if args.start_date:
            cmd += ["--start-date" if args.subset_file else "--start_date", args.start_date]
        if args.end_date:
            cmd += ["--end-date" if args.subset_file else "--end_date", args.end_date]
        run_cmd(cmd, env=lcb_env(args.lcb_dir))
    else:
        log("Generation done. Evaluation skipped; pass --evaluate to run official LCB scoring.")

    log(f"Custom output: {custom_output}")


if __name__ == "__main__":
    main()
