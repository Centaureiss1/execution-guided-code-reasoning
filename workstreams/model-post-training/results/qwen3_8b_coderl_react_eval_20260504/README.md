# Qwen3-8B CodeRL Evaluation Archive

Archived on 2026-05-04 from `/tmp/cs639_grpo` because `/tmp` may be cleared.

This directory stores evaluation outputs only: generated samples, evaluator JSON files, ReAct run logs, final-submit posthoc audits, config/task JSONL files, and console logs. It does not include model weights, Hugging Face caches, venvs, or merged checkpoints.

Useful index files:

- `summary.json`: machine-readable score summary recomputed from this archived copy.
- `MANIFEST.tsv`: file list with relative paths and byte sizes.

## Contents

| directory | contents |
|---|---|
| `evalplus_8b_16k_greedy_base_rl50_rl100_rl150_rl200/` | Non-agentic strict 16k greedy EvalPlus runs for base, rl50, rl100, rl150, rl200. |
| `evalplus_mbpp_greedy/` | Earlier standalone MBPP greedy runs for base, cp100, cp200, cp300. |
| `lcb_runs_34k32k_temp0p2_diverse50/` | Non-agentic official LCB diverse50 hidden evaluation, 34k/32k, temp 0.2. |
| `lcb_runs_32k30k_greedy_bs32/` | Earlier LCB diverse100 generation-only outputs for base, cp200, cp300, cp400. |
| `lcb_runs_32k30k*`, `lcb_runs_32k30k_greedy_bs4/` | Empty/partial early LCB scratch directories, kept for provenance. |
| `agentic_cp200_evalplus_lcb50/` | ReAct cp200 EvalPlus/LCB logs, tasks/configs, HumanEval+ official posthoc, LCB official hidden posthoc. |
| `agentic_cp200/` | Earlier agentic cp200 setup artifacts, including KodCode Light RL 10k task JSONL and vLLM log. |

## Key Results

### ReAct cp200

| benchmark | metric /口径 | result |
|---|---|---:|
| HumanEval | ReAct runner public/base repair@3 | 159/164 = 96.95% |
| HumanEval | official EvalPlus base posthoc | 160/164 = 97.56% |
| HumanEval+ | official EvalPlus plus posthoc | 141/164 = 85.98% |
| MBPP | ReAct runner base repair@3 | 355/378 = 93.92% |
| MBPP+ | EvalPlus plus audit from ReAct logs, missing audits counted as fail | 294/378 = 77.78% |
| LCB diverse50 | ReAct public-test repair@3 | 41/50 = 82.00% |
| LCB diverse50 | official hidden pass@1 posthoc | 36/50 = 72.00% |
| LCB diverse50 temp=0.2 | ReAct public-test repair@3 | 43/50 = 86.00% |
| LCB diverse50 temp=0.2 | official hidden pass@1 posthoc | 34/50 = 68.00% |

Files:

- HumanEval+ posthoc: `agentic_cp200_evalplus_lcb50/official_evalplus_humaneval/react_cp200_humaneval_final_submits_eval_results.json`
- LCB hidden posthoc: `agentic_cp200_evalplus_lcb50/official_lcb_eval/react_cp200_lcb50_official_subset_score_summary.json`
- LCB temp=0.2 hidden posthoc: `agentic_cp200_evalplus_lcb50/official_lcb_eval_temp0p2/react_cp200_lcb50_temp0p2_official_subset_score_summary.json`
- ReAct runs: `agentic_cp200_evalplus_lcb50/logs/*/runs.jsonl`

### Non-Agentic EvalPlus, Strict 16k Greedy

| model | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---:|---:|---:|---:|
| base | 141/164 = 85.98% | 132/164 = 80.49% | 331/378 = 87.57% | 283/378 = 74.87% |
| rl50 | 140/164 = 85.37% | 133/164 = 81.10% | 333/378 = 88.10% | 285/378 = 75.40% |
| rl100 | 141/164 = 85.98% | 132/164 = 80.49% | 333/378 = 88.10% | 287/378 = 75.93% |
| rl150 | 142/164 = 86.59% | 131/164 = 79.88% | 327/378 = 86.51% | 277/378 = 73.28% |
| rl200 | 141/164 = 85.98% | 128/164 = 78.05% | 332/378 = 87.83% | 277/378 = 73.28% |

Files are under `evalplus_8b_16k_greedy_base_rl50_rl100_rl150_rl200/<model>/samples/thinking/`.

### Standalone MBPP Greedy

This is an earlier MBPP-only run with a different run directory/usage from the strict 16k greedy archive above.

| model | MBPP | MBPP+ |
|---|---:|---:|
| base | 336/378 = 88.89% | 291/378 = 76.98% |
| cp100 | 342/378 = 90.48% | 292/378 = 77.25% |
| cp200 | 338/378 = 89.42% | 287/378 = 75.93% |
| cp300 | 171/378 = 45.24% | 150/378 = 39.68% |

Files are under `evalplus_mbpp_greedy/<model>/samples/thinking/`.

### Non-Agentic LCB Diverse50, Official Hidden

| model | pass@1 |
|---|---:|
| base | 15/50 = 30.00% |
| cp50 | 14/50 = 28.00% |
| cp100 | 13/50 = 26.00% |
| cp150 | 13/50 = 26.00% |
| cp200 | 15/50 = 30.00% |

Files are under `lcb_runs_34k32k_temp0p2_diverse50/<model>/`.

### LCB Diverse100 Generation-Only

`lcb_runs_32k30k_greedy_bs32/` contains generated custom outputs but no official score summaries:

| model | records |
|---|---:|
| base | 100 |
| cp200 | 100 |
| cp300 | 100 |
| cp400 | 64 |

## Notes

- ReAct LCB `41/50` is a public-test repair score; the comparable official hidden number is `36/50`.
- ReAct LCB temp=0.2 has higher public repair (`43/50`) but lower official hidden pass (`34/50`) than temp=0.6.
- ReAct HumanEval runner score `159/164` and official EvalPlus base posthoc `160/164` differ by one task because they use different test harnesses.
- MBPP ReAct plus audit has 376 audited tasks in the log; the table counts the two missing audits as failures, hence `294/378`.
