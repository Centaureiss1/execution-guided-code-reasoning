# Reproducibility and environment notes

## Reproducibility level

This repository is best described as a **complete working snapshot with inspectable evidence**, not a one-command reproduction package.

It contains:

- data preparation, training, reward, evaluation, and agent-runner source code;
- YAML configurations and CLI examples;
- selected processed datasets and task subsets;
- generated benchmark samples and full agent traces;
- official evaluator outputs, summary files, and console logs.

It does not contain:

- Qwen, DeepSeek, or other model weights;
- LoRA checkpoints or merged model directories;
- Hugging Face caches;
- the original Python environments;
- the original GPU machines or exact CUDA driver images.

## Environment layers

### `react-bench`

The agent runner targets Python 3.10+ and keeps its core package on the standard library. Optional features add dependencies:

```bash
cd workstreams/agent-benchmark
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[openai,evalplus,prepare]'
```

The original CLI examples and data formats are documented in the [workstream README](../workstreams/agent-benchmark/README.md).

### Post-training experiments

The experiment scripts assume a Linux GPU environment and packages including:

- `datasets`
- `transformers`
- `vllm`
- `evalplus`
- `ms-swift`
- PyTorch with CUDA and `bfloat16` support
- FlashAttention for the 8B configuration

Exact version pins were not captured in the original repositories. Reproduction should therefore record a new lockfile or container image before rerunning experiments.

## Known portability work

Several original scripts and YAML files embed paths such as `/home/brui/cs639_final`, `/workspace/cs639_final`, or `/tmp/cs639_grpo`. These are preserved because the workstreams are provenance snapshots.

Before attempting a new run:

1. replace absolute roots with CLI arguments, environment variables, or paths derived from the repository root;
2. download model weights into a documented external location;
3. verify the upstream dataset versions and redistribution terms;
4. pin Python, CUDA, PyTorch, vLLM, EvalPlus, and `ms-swift` versions;
5. run a one-task smoke test before scheduling training or full evaluation;
6. record model revision, seed, decoding settings, context length, and evaluator version with every output.

## Logical reproduction order

The 1.7B workflow documented by the original project is:

```text
prepare SFT data
    -> run LoRA SFT
    -> merge the SFT adapter
    -> build GRPO candidate pool
    -> score multiple completions by execution
    -> construct difficulty tiers / hard subset
    -> run GRPO with registered custom rewards
    -> merge an evaluation checkpoint if needed
    -> generate HumanEval and MBPP samples
    -> run EvalPlus
```

The 8B workflow represented by the later scripts is:

```text
prepare KodCode/OpenCodeInstruct pool
    -> score Qwen3-8B thinking completions
    -> select empirical hard cases
    -> direct GRPO with execution-gated format reward
    -> one-shot EvalPlus and LCB evaluation
    -> serve checkpoint through a compatible model endpoint
    -> run react-bench for up to three turns
    -> evaluate final submissions on plus / hidden tests
```

## Evaluation semantics

To reproduce a number, match all of the following—not just the model name:

- checkpoint and adapter/base relationship;
- chat template and thinking mode;
- temperature, top-p, top-k, and maximum output length;
- context window and prompt compaction settings;
- one-shot versus multi-turn protocol;
- public runner tests versus EvalPlus plus tests or LCB hidden tests;
- missing-task policy and denominator;
- evaluator package version.

The most reliable frozen record is the [Qwen3-8B archive](../workstreams/model-post-training/results/qwen3_8b_coderl_react_eval_20260504/), which bundles its configs, task subsets, generated outputs, traces, official evaluator outputs, summary, and manifest.

## Safety note

Both training rewards and benchmark adapters execute model-generated Python. The implementation uses subprocesses, timeouts, temporary directories, and process-group cleanup, but this is **not equivalent to a hardened container or VM sandbox**. Run untrusted generations only inside an appropriately isolated environment with restricted filesystem, process, and network access.

## Snapshot verification performed during organization

The portfolio assembly was checked without changing the preserved workstream code:

```text
working-tree copy comparison  passed (excluding .git and .DS_Store)
JSON parsing                  94/94 files passed
Python AST parsing            63/63 files passed under Python 3.12
react-bench unit tests         61/61 passed under Python 3.12
portfolio Markdown links      passed
high-confidence secret scan   no private-key or provider-token match
```

This validation establishes snapshot integrity and basic source health. It does not claim that heavyweight training and evaluation jobs can run on the current machine.
