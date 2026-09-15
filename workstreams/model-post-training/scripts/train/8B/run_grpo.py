"""Launch direct GRPO training for Qwen3-8B.

Use this Python launcher instead of `swift rlhf` so the custom gated code
reward can be registered in ms-swift before training starts.
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import importlib.metadata
import os
import sys
import time
from pathlib import Path


_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))


def log_step(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def require_flash_attn() -> None:
    """Fail early when config asks for FlashAttention2 but the package is absent."""
    log_step("Checking flash_attn availability")
    if importlib.util.find_spec("flash_attn") is not None:
        log_step("flash_attn is available")
        return
    raise SystemExit(
        "flash_attn is required by scripts/train/8B/grpo_config.yaml "
        "(attn_impl: flash_attn), but it is not installed in this environment.\n"
        "Install FlashAttention2 in the active uv environment before training, "
        "then verify with:\n"
        "  /workspace/.venvs/cs639-grpo/bin/python -c \"import flash_attn\""
    )


def log_dependency_versions() -> None:
    """Record package versions in the training log for reproducible debugging."""
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("ms-swift", "ms-swift"),
        ("vllm", "vllm"),
        ("flash-attn", "flash-attn"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("accelerate", "accelerate"),
    ]
    versions: dict[str, str] = {}
    for label, package_name in packages:
        try:
            versions[label] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "<not installed>"
    log_step("Dependency versions:\n" + json.dumps(versions, indent=2, ensure_ascii=False))


def patch_peft_tensor_parallel_import() -> None:
    """Patch a PEFT 0.19.x / Transformers 4.57.x resume-time import mismatch.

    PEFT imports EmbeddingParallel while loading LoRA adapters even when tensor
    parallelism is not active. Some Transformers builds no longer expose that
    class. Single-GPU training does not use the class, so a ColwiseParallel
    alias is enough to let the no-TP path continue.
    """
    try:
        import transformers.integrations.tensor_parallel as tensor_parallel
    except Exception as exc:
        log_step(f"Skipping PEFT tensor-parallel compatibility patch: {exc!r}")
        return

    if hasattr(tensor_parallel, "EmbeddingParallel"):
        log_step("PEFT tensor-parallel compatibility patch not needed")
        return

    tensor_parallel.EmbeddingParallel = tensor_parallel.ColwiseParallel
    log_step("Patched transformers.integrations.tensor_parallel.EmbeddingParallel for PEFT resume")


def load_config_args(config_path: Path, *, steps_override: int | None = None) -> list[str]:
    log_step(f"Loading GRPO config from {config_path}")
    import yaml

    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    if "train_type" in cfg:
        train_type = cfg.pop("train_type")
        cfg.setdefault("tuner_type", train_type)
        log_step(f"Mapped legacy config key train_type={train_type!r} -> tuner_type={cfg['tuner_type']!r}")

    if steps_override is not None:
        cfg["max_steps"] = steps_override
        log_step(f"Overriding max_steps -> {steps_override}")

    log_step("Config loaded:\n" + json.dumps(cfg, indent=2, ensure_ascii=False))

    rlhf_args: list[str] = []
    for key, value in cfg.items():
        arg_name = f"--{key}"
        if isinstance(value, list):
            rlhf_args += [arg_name] + [str(item) for item in value]
        elif isinstance(value, bool):
            rlhf_args += [arg_name, "true" if value else "false"]
        else:
            rlhf_args += [arg_name, str(value)]
    return rlhf_args


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps from grpo_config.yaml, for example --steps 5.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Override config file path. Defaults to scripts/train/8B/grpo_config.yaml.",
    )
    args, remaining = parser.parse_known_args()

    log_step("Starting Qwen3-8B GRPO launcher")
    log_step(f"Python executable: {sys.executable}")
    log_step(f"Working directory: {Path.cwd()}")
    log_step(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    log_step(f"CUDA_HOME={os.environ.get('CUDA_HOME', '<unset>')}")
    log_step(f"LOG_LEVEL={os.environ.get('LOG_LEVEL', '<unset>')}")
    log_step(f"VLLM_LOGGING_LEVEL={os.environ.get('VLLM_LOGGING_LEVEL', '<unset>')}")

    log_dependency_versions()
    require_flash_attn()
    patch_peft_tensor_parallel_import()

    log_step("Importing swift reward registry")
    from swift.rewards.orm import orms
    from reward_fn import CodeExecFormatGatedReward

    orms["code_exec_format_gated"] = CodeExecFormatGatedReward
    log_step("Registered reward: code_exec_format_gated -> CodeExecFormatGatedReward")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_step(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    config_path = Path(args.config) if args.config else _DIR / "grpo_config.yaml"
    rlhf_args = load_config_args(config_path, steps_override=args.steps)
    rlhf_args += remaining

    log_step("Final ms-swift rlhf args:\n" + " ".join(rlhf_args))
    log_step("Importing swift.pipelines.train.rlhf.rlhf_main")
    from swift.pipelines.train.rlhf import rlhf_main

    log_step("Starting ms-swift rlhf_main")
    rlhf_main(rlhf_args)
    log_step("ms-swift rlhf_main finished")


if __name__ == "__main__":
    main()
