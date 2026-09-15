# CS639 Final Project Repository Guide

[中文版本](README_zh.md)

## Overview

This repository is a code-generation experiment centered on `Qwen3-1.7B`. The main workflow is:

```text
data preparation -> SFT -> merged SFT model -> GRPO pool -> offline scoring/tiering
-> GRPO training -> HumanEval/MBPP evaluation -> result snapshots
```

The repo also includes local baseline evaluation for `Qwen2.5-Coder-1.5B-Instruct` and `DeepSeek-R1-Distill-Qwen-1.5B`, plus historical evaluation snapshots stored for reference.

## Repository Map

| Path | Purpose |
| --- | --- |
| `scripts/data/` | Dataset download, filtering, decontamination, pool construction, and offline scoring |
| `scripts/train/1.7B/` | SFT launcher, GRPO launcher, reward definitions, and YAML configs |
| `scripts/eval/` | Sample generation and EvalPlus wrappers |
| `data/processed/` | Derived JSONL datasets and GRPO training variants |
| `models/` | Local model weights, LoRA checkpoints, merged models, and GRPO runs |
| `results/` | Main evaluation outputs produced by the active eval scripts |
| `scripts/eval_results/` | Older or ad hoc result snapshots kept under `scripts/`; these are not executable scripts |
| `logs/` | Log folders for experiments |
| `CS639 Proposal.pdf` | Original project proposal |

## Stage 0: Environment and Dependencies

The active scripts assume a local Python environment with at least these packages:

- `datasets`
- `transformers`
- `vllm`
- `evalplus`
- `ms-swift`

Important repo assumptions:

- Most scripts hardcode the repo root as `/home/brui/cs639_final`.
- Training and inference are set up for local GPU execution with `bfloat16`.
- The Qwen3 training configs use `attn_impl: sdpa` instead of FlashAttention.
- Local model directories are expected to exist under `models/`, especially:
  - `models/Qwen3-1.7B`
  - `models/Qwen3-1.7B/phase1_sft_merged`
  - `models/Qwen2.5-Coder-1.5B-Instruct`
  - `models/DeepSeek-R1-Distill-Qwen-1.5B`

## Pipeline Stages

### Stage 1: Dataset Inspection Helper

**Script:** `scripts/data/download_data.py`

This is a small helper script, not a required pipeline stage. It downloads `ise-uiuc/Magicoder-OSS-Instruct-75K` into `data/raw/` and prints the dataset schema plus the first example so you can inspect field names before writing preprocessing code.

Typical use:

```bash
python scripts/data/download_data.py
```

Inputs and outputs:

- Input: `Magicoder-OSS-Instruct-75K` from Hugging Face
- Output: cached dataset files under `data/raw/`
- Required for reproduction: No

### Stage 2: Build the SFT Training Set

**Script:** `scripts/data/prepare_sft.py`

This script creates `data/processed/sft_train.jsonl`, the SFT chat dataset used in Phase 1 training. It combines two upstream sources:

- `nvidia/OpenCodeInstruct`
- `ise-uiuc/Magicoder-OSS-Instruct-75K`

What it does:

- Filters OpenCodeInstruct by `average_test_score >= 0.9`
- Filters Magicoder to Python-only samples
- Removes empty inputs/outputs
- Drops overly long examples and very short answers
- Deduplicates by prompt text
- Samples a fixed mix:
  - 50,000 OpenCodeInstruct records
  - 20,000 Magicoder records
- Shuffles the merged dataset with a fixed seed

Current output:

- `data/processed/sft_train.jsonl`
- Size in this repo snapshot: about **70,000** rows
- Record format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Typical use:

```bash
python scripts/data/prepare_sft.py
```

### Stage 3: Run SFT, Then Merge the Adapter

**Runner:** `scripts/train/1.7B/run_sft.sh`  
**Config:** `scripts/train/1.7B/sft_config.yaml`

This stage trains a LoRA SFT model on top of `models/Qwen3-1.7B` and writes checkpoints under `models/Qwen3-1.7B/phase1_sft/`.

Key config facts:

- Base model: `models/Qwen3-1.7B`
- Dataset: `data/processed/sft_train.jsonl`
- Training type: LoRA
- `loss_scale: ignore_empty_think`
- Output dir: `models/Qwen3-1.7B/phase1_sft`

Typical use:

```bash
bash scripts/train/1.7B/run_sft.sh
```

Important handoff detail:

- Later steps do **not** consume the SFT adapter directory directly.
- `score_and_tier.py` and the default SFT-based GRPO config both assume a merged full model at `models/Qwen3-1.7B/phase1_sft_merged`.
- The repo does not include a dedicated Phase 1 merge script, so this merge is a manual bridge step.

Generic merge example, based on the same merge logic used by `scripts/eval/run_eval.py`:

```bash
python -m swift.cli.merge_lora \
  --model models/Qwen3-1.7B \
  --adapters models/Qwen3-1.7B/phase1_sft/<run>/checkpoint-<step> \
  --output_dir models/Qwen3-1.7B/phase1_sft_merged \
  --torch_dtype bfloat16
```

### Stage 4: Build the GRPO Candidate Pool

**Script:** `scripts/data/build_grpo_pool.py`

This script builds a unified GRPO candidate pool in JSONL format. It is the first major bridge from supervised data into RL-style training data.

Upstream sources:

- `OpenCodeInstruct`
- `KodCode/KodCode-Light-RL-10K`

What it does:

- Samples high-quality OpenCodeInstruct problems
- Converts OpenCodeInstruct `unit_tests` JSON strings into bare `assert` lists
- Converts KodCode pytest-style tests into bare `assert` lists
- Removes problems overlapping with:
  - HumanEval
  - all MBPP splits
  - the SFT training set
- Deduplicates across OpenCodeInstruct and KodCode
- Wraps each task into a chat-style prompt with a system message

Current output:

- `data/processed/grpo_pool.jsonl`
- Size in this repo snapshot: about **49,825** rows
- Record format:

```json
{"messages": [...], "tests": ["assert ..."], "source": "...", "question_id": "..."}
```

Typical use:

```bash
python scripts/data/build_grpo_pool.py
```

### Stage 5: Offline Scoring and Tiering

**Script:** `scripts/data/score_and_tier.py`

This script uses the merged SFT model to score each GRPO candidate by actual execution success, then builds a filtered training set for GRPO.

What it does:

- Loads `data/processed/grpo_pool.jsonl`
- Uses `models/Qwen3-1.7B/phase1_sft_merged` for vLLM batch generation
- Generates multiple completions per problem
- Extracts Python code blocks from the model output
- Runs the code against the task's `assert` tests in an isolated subprocess
- Computes `pass_rate` for each problem
- Writes resumable checkpoints into `grpo_pool_scored.jsonl`
- Builds a tiered GRPO training set in `grpo_train.jsonl`

Current outputs:

- `data/processed/grpo_pool_scored.jsonl` with about **49,825** rows
- `data/processed/grpo_train.jsonl` with about **10,283** rows

Supported CLI options:

- `--score-only`: score the pool without producing the final tiered dataset
- `--tier-only`: skip model inference and rebuild tiers from an existing scored file
- `--n-samples`: change the number of completions used per problem

Typical use:

```bash
python scripts/data/score_and_tier.py
python scripts/data/score_and_tier.py --score-only
python scripts/data/score_and_tier.py --tier-only
python scripts/data/score_and_tier.py --n-samples 4
```

### Stage 5.5: Experimental GRPO Dataset Variants

The repo currently contains several extra GRPO-related datasets in `data/processed/`, but the scripts that created these specific variants are not tracked here:

| File | Current size | What it appears to represent |
| --- | --- | --- |
| `grpo_train.jsonl` | 10,283 | Default tiered set produced by `score_and_tier.py` |
| `grpo_train_curriculum.jsonl` | 10,283 | Curriculum-style experimental variant |
| `grpo_train_shuffle.jsonl` | 10,283 | Shuffled experimental variant |
| `grpo_train_hard.jsonl` | 5,018 | Hard subset used by the active GRPO configs |
| `grpo_mbpp_train.jsonl` | 374 | Separate MBPP-derived GRPO-style set |

Two practical consequences:

- `scripts/train/1.7B/grpo_config.yaml` currently points to `grpo_train_hard.jsonl`, not `grpo_train.jsonl`.
- Reproducing those variant datasets exactly would require logic that is not currently committed as a script in this repo.

### Stage 6: GRPO Training

**Runner:** `scripts/train/1.7B/run_grpo.py`  
**Rewards:** `scripts/train/1.7B/reward_fn.py`  
**Configs:**  
- `scripts/train/1.7B/grpo_config.yaml`
- `scripts/train/1.7B/grpo_config_base_hard.yaml`

This stage launches GRPO training through `ms-swift`, but with one important twist: the reward functions are custom Python classes, so the training job must be started through `run_grpo.py`, not directly through `swift rlhf`.

`reward_fn.py` defines two reward components:

- `CodeExecReward`: extracts Python code and returns `1.0` only when the code passes all tests
- `CodeFormatReward`: rewards completions that preserve the expected thinking/code-block format

`run_grpo.py`:

- adds the local training directory to `sys.path`
- registers the custom rewards into `swift.rewards.orm.orms`
- loads a YAML config
- optionally overrides `max_steps`
- launches `swift.pipelines.train.rlhf.rlhf_main(...)`

Config split:

- `grpo_config.yaml`
  - starts from `models/Qwen3-1.7B/phase1_sft_merged`
  - currently trains on `data/processed/grpo_train_hard.jsonl`
  - outputs to `models/Qwen3-1.7B/phase2_grpo_hard`
- `grpo_config_base_hard.yaml`
  - starts from the base `models/Qwen3-1.7B`
  - also trains on `data/processed/grpo_train_hard.jsonl`
  - outputs to `models/Qwen3-1.7B/phase2_grpo_base_hard`

Supported CLI options:

- `--steps`: override `max_steps` from the YAML config
- `--config`: switch between GRPO configs

Typical use:

```bash
python scripts/train/1.7B/run_grpo.py
python scripts/train/1.7B/run_grpo.py --steps 10
python scripts/train/1.7B/run_grpo.py --config scripts/train/1.7B/grpo_config_base_hard.yaml --steps 800
```

### Stage 7: Evaluate the Main Qwen3 Models

**Scripts:**

- `scripts/eval/gen_samples.py`
- `scripts/eval/run_eval.py`

This is the main evaluation path for:

- the base `Qwen3-1.7B` model
- the merged SFT model
- GRPO LoRA checkpoints

How it works:

- `run_eval.py` decides whether the input model path is a full model or a LoRA adapter checkpoint
- if needed, it merges the LoRA adapter into a temporary full model under `results/<exp>/merged_model`
- it calls `gen_samples.py` to generate HumanEval and MBPP solutions
- it then runs `python -m evalplus.evaluate` on the generated JSONL samples

`gen_samples.py` is responsible for:

- formatting prompts with the Qwen chat template
- handling `thinking`, `auto`, and `no_thinking` modes
- extracting the best code block from raw model output
- writing EvalPlus-compatible sample JSONL files

Supported CLI options in `run_eval.py`:

- `--model`
- `--base-model`
- `--output-dir`
- `--thinking`
- `--auto`
- `--skip-gen`
- `--max-new-tokens`

Typical use:

```bash
python scripts/eval/run_eval.py \
  --model models/Qwen3-1.7B \
  --output-dir results/base_v2 \
  --auto

python scripts/eval/run_eval.py \
  --model models/Qwen3-1.7B/phase1_sft_merged \
  --output-dir results/sft_v2 \
  --auto

python scripts/eval/run_eval.py \
  --model models/Qwen3-1.7B/phase2_grpo_hard/<run>/checkpoint-800 \
  --base-model models/Qwen3-1.7B/phase1_sft_merged \
  --output-dir results/grpo_hard_800 \
  --auto
```

### Stage 8: Evaluate Local Baseline Models

**Scripts:**

- `scripts/eval/gen_samples_vllm_generic.py`
- `scripts/eval/run_eval_vllm_generic.py`

These scripts are the baseline evaluation path for local non-Qwen3 models. In the current repo, that mainly means:

- `Qwen2.5-Coder-1.5B-Instruct`
- `DeepSeek-R1-Distill-Qwen-1.5B`

How this path differs from the main Qwen3 evaluation path:

- it uses family-specific prompt formatting and sampling defaults
- it runs in `native` mode rather than the Qwen3 `thinking/auto/no_thinking` modes
- it is intended for full local models, not LoRA adapters

Supported CLI options in `run_eval_vllm_generic.py`:

- `--family`
- `--dataset`
- `--max-model-len`
- `--max-new-tokens`
- `--limit`
- `--skip-gen`

Typical use:

```bash
python scripts/eval/run_eval_vllm_generic.py \
  --model models/Qwen2.5-Coder-1.5B-Instruct \
  --family qwen25_coder \
  --output-dir results/qwen25_coder_1_5b_instruct

python scripts/eval/run_eval_vllm_generic.py \
  --model models/DeepSeek-R1-Distill-Qwen-1.5B \
  --family deepseek_r1_distill \
  --output-dir results/deepseek_r1_distill_qwen_1_5b_auto
```

### Stage 9: Result Directories

The `results/` directory stores the main evaluation outputs created by the active evaluation scripts. In this repo snapshot, the key subdirectories are:

| Directory | Meaning |
| --- | --- |
| `results/base_v2/` | Base `Qwen3-1.7B` evaluation snapshot |
| `results/sft_v2/` | Merged SFT model evaluation snapshot |
| `results/grpo_hard_800/` | GRPO-from-SFT hard-set run evaluated at checkpoint 800 |
| `results/grpo_base_hard_800/` | GRPO-from-base hard-set run evaluated at checkpoint 800 |
| `results/grpo_shuffle_600/` | Shuffle-variant GRPO run evaluated at checkpoint 600 |
| `results/qwen25_coder_1_5b_instruct/` | Local Qwen2.5-Coder baseline evaluation |
| `results/deepseek_r1_distill_qwen_1_5b_auto/` | Local DeepSeek-R1-Distill baseline evaluation |

Two details are easy to miss:

- `results/*/samples/...` contains the generated sample JSONL files and EvalPlus result JSON files.
- `results/*/merged_model/` may be created automatically by `run_eval.py` when it has to merge a LoRA adapter before evaluation.

Historical note:

- `scripts/eval_results/*` contains older evaluation artifacts kept under the `scripts/` tree.
- Treat those as archived reference outputs, not as runnable evaluation entry points.

## Appendix A: Script Index

| Script | Main role | Main inputs | Main outputs | Required in main pipeline? | Typical timing |
| --- | --- | --- | --- | --- | --- |
| `scripts/data/download_data.py` | Quick schema inspection for Magicoder | HF dataset | Cached raw data | No | One-off inspection |
| `scripts/data/prepare_sft.py` | Build the SFT training set | OpenCodeInstruct, Magicoder | `sft_train.jsonl` | Yes | Before Phase 1 SFT |
| `scripts/train/1.7B/run_sft.sh` | Launch SFT | `sft_config.yaml`, base Qwen3 model | LoRA checkpoints in `phase1_sft/` | Yes | Phase 1 training |
| `scripts/data/build_grpo_pool.py` | Build decontaminated GRPO pool | OCI, KodCode, eval fingerprints, SFT prompts | `grpo_pool.jsonl` | Yes | After SFT data prep |
| `scripts/data/score_and_tier.py` | Score pool with merged SFT model and create GRPO train set | `grpo_pool.jsonl`, `phase1_sft_merged` | `grpo_pool_scored.jsonl`, `grpo_train.jsonl` | Yes | Before GRPO |
| `scripts/train/1.7B/reward_fn.py` | Define custom GRPO rewards | Model completions, dataset `tests` | In-memory reward values | Indirectly yes | Imported during GRPO startup |
| `scripts/train/1.7B/run_grpo.py` | Launch GRPO with custom reward registration | GRPO YAML config, reward classes | GRPO checkpoints and logs | Yes | Phase 2 training |
| `scripts/eval/gen_samples.py` | Generate Qwen3 eval samples | Main model path, HumanEval/MBPP | Sample JSONL | Yes | Called by `run_eval.py` |
| `scripts/eval/run_eval.py` | Orchestrate Qwen3 eval and LoRA merge | Model path, optional base model | `results/*` eval outputs | Yes | After training |
| `scripts/eval/gen_samples_vllm_generic.py` | Generate baseline-model eval samples | Baseline model path, model family | Sample JSONL | For baselines only | Baseline evaluation |
| `scripts/eval/run_eval_vllm_generic.py` | Orchestrate baseline eval | Baseline model path, family | `results/*` baseline outputs | For baselines only | Baseline evaluation |

### Key Config Files

These are not scripts, but they matter for understanding the pipeline:

| File | Role |
| --- | --- |
| `scripts/train/1.7B/sft_config.yaml` | Phase 1 SFT config |
| `scripts/train/1.7B/grpo_config.yaml` | SFT-initialized GRPO config on the hard dataset |
| `scripts/train/1.7B/grpo_config_base_hard.yaml` | Base-model GRPO config on the hard dataset |

## Appendix B: Common Gotchas

- **You need a merged SFT model after Phase 1.** The downstream scoring script and the default SFT-based GRPO config expect `models/Qwen3-1.7B/phase1_sft_merged`, not a raw LoRA checkpoint directory.
- **Do not replace `run_grpo.py` with `swift rlhf`.** The Python launcher patches `ms-swift`'s reward registry before training starts; without that patch, the custom reward names in the config will not resolve.
- **`--base-model` is only needed when evaluating a LoRA adapter checkpoint.** If `--model` already points to a full merged model, `run_eval.py` can evaluate it directly.
- **The active GRPO configs point to `grpo_train_hard.jsonl`.** That is different from the default `grpo_train.jsonl` created by `score_and_tier.py`.
- **Many paths are absolute.** If you move the repo, several scripts will need path edits before they run successfully.
- **`scripts/eval_results/*` is historical output, not active code.** Use `scripts/eval/*.py` for current evaluation runs.
