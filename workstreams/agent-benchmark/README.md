# react-bench

Benchmark-oriented Code ReAct runner for EvalPlus, LiveCodeBench-style, and KodCode-style Python tasks.

The v1 loop is intentionally narrow:

```text
<input>...</input>
<think>...</think>
<submit>...</submit>
<obs>status: fail ...</obs>
...
```

It supports multi-turn repair, Azure OpenAI/vLLM/OpenAI-compatible providers, structured JSONL logs, failed-prefix collection, teacher continuation, and SFT trace export.
`<submit>` is required. `<think>` is extracted when a model exposes native thinking text or emits explicit think tags; submit-only outputs are valid.

## Install

The core package uses only the Python standard library.

```bash
pip install -e .
pip install openai  # only needed for API providers
```

## Run

```bash
react-bench run \
  --benchmark evalplus \
  --dataset humaneval \
  --provider vllm \
  --model Qwen3-8B \
  --config configs/compact_repair.yaml
```

Use `--verbose` for per-turn console progress. Add `--print-submit` or `--print-raw` when debugging model output formatting; full raw outputs are always saved in `runs.jsonl`.

The default config uses a 16k generation budget plus forced-submit recovery. If a model hits the output limit before producing `<submit>`, the runner makes a second same-turn call that asks for code only.
For vLLM recovery calls, the provider also requests `enable_thinking=false` through `extra_body` when `recovery_disable_thinking: true`.

For Azure OpenAI, use the Azure deployment name as `--model` and set either `AZURE_OPENAI_ENDPOINT` or `--base-url`:

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com

react-bench run \
  --benchmark livecodebench \
  --scenario codegeneration \
  --tasks path/to/lcb_tasks.jsonl \
  --provider azure_openai \
  --model AZURE_DEPLOYMENT
```

## Local Task JSONL

To prepare a real LiveCodeBench JSONL from Hugging Face:

```bash
pip install "react-bench[prepare]"

react-bench prepare-lcb \
  --version v6 \
  --limit 10 \
  --output data/livecodebench/codegeneration.jsonl
```

Use `--version release_v6` for the full current release. The default `v6` is the smaller recent slice and is faster for debugging.

To prepare KodCode-Light-RL-10K:

```bash
react-bench prepare-kodcode \
  --output data/kodcode/light_rl_10k.jsonl
```

KodCode rows are run as Python module tests: the model submission is saved as `solution.py`, and the dataset's unit tests are run against imports from `solution`. You can run a small smoke test with:

```bash
react-bench run \
  --benchmark kodcode \
  --dataset train \
  --tasks data/kodcode/light_rl_10k.jsonl \
  --provider vllm \
  --model qwen3-4b-awq \
  --base-url http://localhost:8000/v1 \
  --config configs/debug_repair.yaml \
  --limit 5 \
  --verbose \
  --print-submit
```

EvalPlus-like rows can provide explicit tests:

```json
{"task_id":"add","prompt":"Write add(a, b).","entry_point":"add","tests":["assert add(1, 2) == 3"]}
```

Or differential EvalPlus-style fields:

```json
{"task_id":"add","prompt":"def add(a, b):\n","entry_point":"add","canonical_solution":"    return a + b","base_input":[[1,2]],"plus_input":[[-1,1]]}
```

LiveCodeBench-like rows use stdin/stdout cases:

```json
{"question_id":"sum","question_content":"Read two ints and print their sum.","public_test_cases":[{"input":"1 2\n","output":"3\n"}]}
```

KodCode-like rows use module tests:

```json
{"question_id":"inc","question":"Return x + 1.","test":"from solution import inc\n\ndef test_inc():\n    assert inc(1) == 2","test_info":[{"function_declaration":"def inc(x):"}]}
```

## Distill and Export

```bash
react-bench distill \
  --failed-prefixes logs/failed_prefixes.jsonl \
  --teacher-provider azure_openai \
  --teacher-model AZURE_DEPLOYMENT \
  --max-turns 3

react-bench export-sft \
  --teacher-traces logs/teacher_traces.jsonl \
  --output data/sft_traces.jsonl
```

## Current Scope

- Python execution only.
- EvalPlus, LiveCodeBench codegeneration, and KodCode module-test tasks.
- Standard pass@1 should be reported separately from multi-turn `repair@k`.
- Hidden/oracle feedback is supported for distillation only and is recorded in trace metadata.
