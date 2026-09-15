#!/bin/bash
# Phase 1 SFT — Qwen2.5-Coder-1.5B-Instruct LoRA
# 用法: bash scripts/train/q25_1.5b/run_sft.sh

set -e
cd /home/brui/cs639_final

# 屏蔽 nvidia-cdi-refresh 防止 Blackwell CUDA 崩溃
sudo systemctl mask nvidia-cdi-refresh.service nvidia-cdi-refresh.path 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Phase 1 SFT 开始 ==="
swift sft --config scripts/train/q25_1.5b/sft_config.yaml

echo "=== 训练完成 ==="
echo "Checkpoint 保存在: models/Qwen2.5-Coder-1.5B-Instruct/phase1_sft/"
