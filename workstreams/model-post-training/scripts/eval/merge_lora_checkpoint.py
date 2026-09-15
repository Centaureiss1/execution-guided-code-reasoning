"""Merge a Swift/PEFT LoRA checkpoint into a standalone model directory.

This helper mirrors the compatibility patch used by the eval wrappers, so it
works with the local transformers/peft versions that miss EmbeddingParallel.
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path


def patch_tensor_parallel_import() -> None:
    try:
        import transformers.integrations.tensor_parallel as tensor_parallel

        if not hasattr(tensor_parallel, "EmbeddingParallel"):
            tensor_parallel.EmbeddingParallel = tensor_parallel.ColwiseParallel
            print("[merge] Patched transformers.integrations.tensor_parallel.EmbeddingParallel")
    except Exception as exc:
        print(f"[merge] Skipping tensor-parallel compatibility patch: {exc!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--force", action="store_true", help="Remove output-dir before merging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.force and output_dir.exists():
        print(f"[merge] Removing existing output dir: {output_dir}")
        shutil.rmtree(output_dir)
    if (output_dir / "config.json").exists():
        print(f"[merge] Output already exists, skipping: {output_dir}")
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    patch_tensor_parallel_import()
    sys.argv = [
        "swift.cli.merge_lora",
        "--model",
        str(Path(args.base_model).resolve()),
        "--adapters",
        str(Path(args.adapter).resolve()),
        "--output_dir",
        str(output_dir),
        "--torch_dtype",
        args.torch_dtype,
    ]
    runpy.run_module("swift.cli.merge_lora", run_name="__main__")


if __name__ == "__main__":
    main()
