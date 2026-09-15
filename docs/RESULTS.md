# Results and evidence

## Reading guide

The headline table follows Section 4.1 of the project's final written report. Supplementary tables are traced to evaluator outputs stored in the repository. Percentages are pass counts divided by the full benchmark size unless noted otherwise.

Three labels matter:

- **one-shot:** one generated answer per task, without repair feedback;
- **repair@3:** up to three submissions with test feedback between turns;
- **plus / hidden:** stricter tests not exposed as repair observations.

Results with different labels answer different questions and should not be treated as a single leaderboard.

## Final Qwen3-8B results

Source of record: the final project report, Section 4.1, "Main benchmark results."

| Model / setting | HumanEval | HumanEval+ | MBPP | MBPP+ | LiveCodeBench (50 tasks) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 83.5% | 76.8% | 88.9% | 76.9% | 15/50 |
| RL 200 | 86.6% | 81.7% | 89.4% | 75.9% | 15/50 |
| RL 300 | 87.2% | 81.1% | 45.2% | 39.7% | - |
| RL 200 + ReAct runner, up to 3 turns | **97.6%** | **86.0%** | **93.9%** | **77.8%** | **34/50** |

### Evaluation protocol

- First-attempt results use greedy decoding, a 16k total context budget, and a 14k maximum output length.
- The ReAct setting allows up to 16k output tokens per generation and uses a 32k full multi-turn context budget.
- `RL 200 + ReAct` is a test-time repair system using the stable RL checkpoint. It did not receive Agent SFT or Agent RL.
- The agent can use verified feedback from available tests but cannot observe plus or hidden benchmark tests.

### Training-time result

The stable 200-step checkpoint partially confirms the direct-GRPO hypothesis:

- HumanEval increases by 3.1 percentage points, from 83.5% to 86.6%.
- HumanEval+ increases by 4.9 points, from 76.8% to 81.7%.
- MBPP increases by 0.5 points, from 88.9% to 89.4%.
- MBPP+ decreases by 1.0 point, from 76.9% to 75.9%.
- LiveCodeBench remains 15/50.

The later checkpoint is not a successful continuation of that trend. RL 300 reaches 87.2% HumanEval and 81.1% HumanEval+, but MBPP collapses to 45.2% and MBPP+ to 39.7%. The final report attributes this instability to malformed long-thinking outputs, incomplete submissions, and Python syntax errors, with a small training set, aggressive learning rate, no KL regularization, and no explicit syntax or length reward identified as likely contributors.

### Inference-time result

Relative to RL 200 first-attempt evaluation, adding up to three repair turns changes the results as follows:

| Metric | RL 200 | RL 200 + ReAct | Difference |
| --- | ---: | ---: | ---: |
| HumanEval | 86.6% | 97.6% | +11.0 percentage points |
| HumanEval+ | 81.7% | 86.0% | +4.3 percentage points |
| MBPP | 89.4% | 93.9% | +4.5 percentage points |
| MBPP+ | 75.9% | 77.8% | +1.9 percentage points |
| LiveCodeBench | 15/50 | 34/50 | +19 tasks (+38.0 points) |

This is a **system-level test-time scaling comparison**, not a pure model comparison: the repair setup changes the interaction protocol, context budget, number of generation opportunities, and access to test feedback. The smaller improvement on plus tests supports the report's conclusion that repair is most effective when the environment exposes diagnostic feedback relevant to the failure.

## Additional archived Qwen3-8B variants

The [`2026-05-04 evaluation archive`](../workstreams/model-post-training/results/qwen3_8b_coderl_react_eval_20260504/) contains additional checkpoint sweeps, decoding settings, public-runner repair scores, and post-hoc audits. Its [`summary.json`](../workstreams/model-post-training/results/qwen3_8b_coderl_react_eval_20260504/summary.json) is useful for studying those variants, but it is not the source of the portfolio's final headline table. The final written report was compiled later and selected the results shown above.

## Qwen3-1.7B main experiments

The table below is recomputed from the stored EvalPlus result JSON files under [`results/`](../workstreams/model-post-training/results/). All listed rows use the repository's `auto` evaluation path.

| Experiment | HumanEval | HumanEval+ | MBPP | MBPP+ |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3-1.7B | 124/164 (75.61%) | 115/164 (70.12%) | 289/378 (76.46%) | 253/378 (66.93%) |
| SFT | 108/164 (65.85%) | 101/164 (61.59%) | 248/378 (65.61%) | 214/378 (56.61%) |
| GRPO from SFT, hard set, step 800 | 119/164 (72.56%) | 114/164 (69.51%) | 254/378 (67.20%) | 220/378 (58.20%) |
| GRPO from base, hard set, step 800 | 129/164 (78.66%) | 120/164 (73.17%) | 295/378 (78.04%) | 252/378 (66.67%) |
| GRPO shuffled variant, step 600 | 113/164 (68.90%) | 102/164 (62.20%) | 243/378 (64.29%) | 209/378 (55.29%) |

The strongest overall 1.7B snapshot is the hard-set GRPO run initialized from the base model. It improves HumanEval and HumanEval+ over the base snapshot and narrowly improves standard MBPP, while MBPP+ is effectively flat. SFT initialization underperforms the base in this experiment series.

Direct evidence directories:

- [`base_v2`](../workstreams/model-post-training/results/base_v2/)
- [`sft_v2`](../workstreams/model-post-training/results/sft_v2/)
- [`grpo_hard_800`](../workstreams/model-post-training/results/grpo_hard_800/)
- [`grpo_base_hard_800`](../workstreams/model-post-training/results/grpo_base_hard_800/)
- [`grpo_shuffle_600`](../workstreams/model-post-training/results/grpo_shuffle_600/)

## Additional 1.5B baselines and SFT snapshots

| Experiment | HumanEval | HumanEval+ | MBPP | MBPP+ |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-1.5B-Instruct | 95/164 (57.93%) | 88/164 (53.66%) | 225/378 (59.52%) | 196/378 (51.85%) |
| DeepSeek-R1-Distill-Qwen-1.5B | 104/164 (63.41%) | 97/164 (59.15%) | 215/378 (56.88%) | 188/378 (49.74%) |
| Qwen2.5 1.5B SFT, 1,560-step parser-fix run | 91/164 (55.49%) | 79/164 (48.17%) | 201/378 (53.17%) | 183/378 (48.41%) |
| Qwen2.5 1.5B SFT 50k, step 9,360 | 88/164 (53.66%) | 81/164 (49.39%) | 202/378 (53.44%) | 178/378 (47.09%) |

Some intermediate SFT snapshots contain only HumanEval outputs; they remain in the archive but are omitted from this paired table.

## Evidence hierarchy

When numbers disagree, use this order:

1. the final written report for the portfolio's headline result selection;
2. official evaluator result JSON or hidden-test summary for an individual run;
3. the machine-readable 8B `summary.json` for additional archived variants;
4. `runs.jsonl` for runner-level behavior and audit reconstruction;
5. console logs and generated-only output files.

The 8B archive includes a [`MANIFEST.tsv`](../workstreams/model-post-training/results/qwen3_8b_coderl_react_eval_20260504/MANIFEST.tsv) with relative paths and byte sizes for provenance.

## Limitations

- The evaluation sets are small enough that a few tasks can move percentages materially.
- Some comparisons change sampling, context length, prompt format, or evaluation harness.
- Process-based execution reduces accidental interference but is not a hardened security sandbox for untrusted code.
- The archive does not include model weights, so stored generations and evaluator outputs are the primary evidence.
- A subset score should not be presented as a full benchmark score.
