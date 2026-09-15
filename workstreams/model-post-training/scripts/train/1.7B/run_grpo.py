"""Launch GRPO training for Qwen3-1.7B.

必须通过 Python 启动（不能用 `swift rlhf` 命令行），
原因：需要在 Python 进程内 patch ms-swift orms dict 注入自定义 reward 类。

Usage:
    python scripts/train/1.7B/run_grpo.py              # 全量（config 里的 max_steps=400）
    python scripts/train/1.7B/run_grpo.py --steps 10   # 快速测试（10 steps）
    python scripts/train/1.7B/run_grpo.py --steps 800  # 续训
"""

import argparse
import os
import sys

# ── 将脚本目录加入 path，使 reward_fn.py 可导入 ─────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

# ── 注册自定义 reward 到 ms-swift orms dict ───────────────────────────────────
# 必须在 import swift 训练模块之前完成，否则 config 里的 reward_funcs 名称找不到
from swift.rewards.orm import orms
from reward_fn import CodeExecReward, CodeFormatReward

orms["code_exec"]   = CodeExecReward
orms["code_format"] = CodeFormatReward

# ── 显存优化环境变量 ──────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── 解析 --steps 参数 ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--steps", type=int, default=None,
                    help="Override max_steps in grpo_config.yaml (e.g. --steps 10 for quick test)")
parser.add_argument("--config", default=None,
                    help="Override config file (default: grpo_config.yaml)")
args, remaining = parser.parse_known_args()

# ── 读 YAML config 展开为 --key value 参数列表 ───────────────────────────────
import yaml
config_path = args.config or os.path.join(_DIR, "grpo_config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

rlhf_args = []
for k, v in cfg.items():
    if isinstance(v, list):
        rlhf_args += [f"--{k}"] + [str(x) for x in v]
    elif isinstance(v, bool):
        if v:
            rlhf_args.append(f"--{k}")
            rlhf_args.append("true")
        else:
            rlhf_args.append(f"--{k}")
            rlhf_args.append("false")
    else:
        rlhf_args += [f"--{k}", str(v)]

rlhf_args += remaining

if args.steps is not None:
    rlhf_args += ["--max_steps", str(args.steps)]
    print(f"[run_grpo] Overriding max_steps → {args.steps}")

# ── 启动训练 ──────────────────────────────────────────────────────────────────
from swift.pipelines.train.rlhf import rlhf_main
rlhf_main(rlhf_args)
