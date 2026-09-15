# Repository map and provenance

## Unified portfolio layer

| Path | Purpose |
| --- | --- |
| [`README.md`](../README.md) | English public-facing project narrative and navigation |
| [`README_zh.md`](../README_zh.md) | Chinese project narrative and navigation |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | End-to-end component and data-flow explanation |
| [`RESULTS.md`](RESULTS.md) | Evidence-backed metrics and limitations |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Environment assumptions and rerun guidance |
| [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md) | Items to resolve before making the repository public |
| [`.gitattributes`](../.gitattributes) | Review-friendly treatment for generated and large artifacts |
| [`.gitignore`](../.gitignore) | Excludes new local environments, caches, weights, and secrets |

## Workstream 1: agent benchmark

Path: [`workstreams/agent-benchmark/`](../workstreams/agent-benchmark/)

Snapshot facts:

- original repository: `https://github.com/Centaureiss1/cs639_agentic.git`
- branch: `main`
- commit: `be993282d38081cacb9d724ef72577c1ef254b2c`
- last recorded commit date: `2026-05-04T15:43:08Z`
- copied working-tree files: 54
- source/config/documentation lines (`.py`, `.sh`, `.yaml`, `.md`): approximately 4,474

| Path | Purpose |
| --- | --- |
| [`react_bench/agent/`](../workstreams/agent-benchmark/react_bench/agent/) | Multi-turn runner orchestration |
| [`react_bench/benchmarks/`](../workstreams/agent-benchmark/react_bench/benchmarks/) | EvalPlus, LiveCodeBench-style, and KodCode adapters |
| [`react_bench/execution/`](../workstreams/agent-benchmark/react_bench/execution/) | Python execution support |
| [`react_bench/providers/`](../workstreams/agent-benchmark/react_bench/providers/) | OpenAI-compatible, Azure OpenAI, and vLLM providers |
| [`react_bench/protocol/`](../workstreams/agent-benchmark/react_bench/protocol/) | Schema, parsing, rendering, and prompt compaction |
| [`react_bench/logging/`](../workstreams/agent-benchmark/react_bench/logging/) | Console and structured JSONL run logging |
| [`react_bench/export/`](../workstreams/agent-benchmark/react_bench/export/) | Failed-prefix and SFT export support |
| [`configs/`](../workstreams/agent-benchmark/configs/) | Compact and debug repair configurations |
| [`tests/`](../workstreams/agent-benchmark/tests/) | Unit tests covering runner, adapters, providers, executor, parser, logging, preparation, and CLI |
| [`CS639_Final_Proposal.pdf`](../workstreams/agent-benchmark/CS639_Final_Proposal.pdf) | Preserved project proposal |

## Workstream 2: model post-training

Path: [`workstreams/model-post-training/`](../workstreams/model-post-training/)

Snapshot facts:

- original repository: `https://github.com/Centaureiss1/cs639_final.git`
- branch: `master`
- commit: `d0de212bc80632bce38289c5c2e25032a4616f4b`
- last recorded commit date: `2026-05-04T20:13:09Z`
- copied working-tree files: 230
- source/config/documentation lines (`.py`, `.sh`, `.yaml`, `.md`): approximately 7,692

| Path | Purpose |
| --- | --- |
| [`scripts/data/`](../workstreams/model-post-training/scripts/data/) | Dataset preparation, decontamination, scoring, and tiering |
| [`scripts/train/1.7B/`](../workstreams/model-post-training/scripts/train/1.7B/) | Qwen3-1.7B SFT/GRPO launchers, configs, and rewards |
| [`scripts/train/8B/`](../workstreams/model-post-training/scripts/train/8B/) | Qwen3-8B direct GRPO configuration and gated reward |
| [`scripts/train/q25_1.5b/`](../workstreams/model-post-training/scripts/train/q25_1.5b/) | Qwen2.5-Coder 1.5B SFT setup |
| [`scripts/eval/`](../workstreams/model-post-training/scripts/eval/) | EvalPlus and LiveCodeBench generation/evaluation workflows |
| [`data/eval/`](../workstreams/model-post-training/data/eval/) | Frozen LiveCodeBench evaluation subsets |
| [`data/processed/`](../workstreams/model-post-training/data/processed/) | Processed candidate pools, scored shards, and training subsets |
| [`evals/`](../workstreams/model-post-training/evals/) | Earlier EvalPlus checkpoint snapshots |
| [`results/`](../workstreams/model-post-training/results/) | 1.5B/1.7B evaluations and the complete 8B archive |

## Preservation policy

The new portfolio uses an additive wrapper rather than rewriting the original projects.

- Original working-tree content and relative layout are preserved under the two workstream folders.
- Original `.git` directories are excluded to avoid nested repositories and accidental publication of repository internals.
- `.DS_Store` files are excluded as operating-system metadata.
- Source repositories remain untouched beside this unified directory.
- A recursive comparison was used after copying to verify equality under those exclusions.

At portfolio creation time, the two snapshots contained **284 files** and occupied approximately **443 MB** together. Most of that size is processed JSONL data and generated benchmark output, not source code.

## Artifact classes

| Class | Examples | Interpretation |
| --- | --- | --- |
| Source | `.py`, `.sh` | Implemented pipeline and framework logic |
| Configuration | `.yaml`, `pyproject.toml` | Frozen experiment or package settings |
| Processed data | `data/processed/*.jsonl` | Derived training/scoring snapshots |
| Generated samples | `results/**/samples/*` | Raw model outputs used for evaluation |
| Evaluator outputs | `*_eval_results.json`, `*_score_summary.json` | Primary metric evidence |
| Agent traces | `runs.jsonl`, `failed_prefixes.jsonl` | Turn-level execution and repair behavior |
| Operational logs | `.log`, `.pid`, `MANIFEST.tsv` | Provenance and run diagnostics |
| Documentation | `README*.md`, proposal PDF | Original and portfolio-level explanation |

## Suggested reviewer path

For a ten-minute review:

1. read the root [README](../README.md);
2. inspect the [architecture](ARCHITECTURE.md);
3. review the [results and caveats](RESULTS.md);
4. open the 8B archive's [`summary.json`](../workstreams/model-post-training/results/qwen3_8b_coderl_react_eval_20260504/summary.json);
5. inspect [`ReActRunner`](../workstreams/agent-benchmark/react_bench/agent/runner.py) and the [8B gated reward](../workstreams/model-post-training/scripts/train/8B/reward_fn.py);
6. use the workstream READMEs for deeper implementation details.
