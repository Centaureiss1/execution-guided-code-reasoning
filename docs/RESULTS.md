# Results and evidence

## Reading guide

This report summarizes only metrics that can be traced to evaluator outputs stored in the repository. Percentages are pass counts divided by the full benchmark size unless noted otherwise.

Three labels matter:

- **one-shot:** one generated answer per task, without repair feedback;
- **repair@3:** up to three submissions with test feedback between turns;
- **plus / hidden:** stricter tests not exposed as repair observations.

Results with different labels answer different questions and should not be treated as a single leaderboard.

## Qwen3-8B archive

Source of record: [`summary.json`](../workstreams/model-post-training/results/qwen3_8b_coderl_react_eval_20260504/summary.json) and the accompanying [archive README](../workstreams/model-post-training/results/qwen3_8b_coderl_react_eval_20260504/README.md).

### One-shot EvalPlus, strict 16k greedy

| Checkpoint | HumanEval | HumanEval+ | MBPP | MBPP+ |
| --- | ---: | ---: | ---: | ---: |
| Base | 141/164 (85.98%) | 132/164 (80.49%) | 331/378 (87.57%) | 283/378 (74.87%) |
| RL 50 | 140/164 (85.37%) | 133/164 (81.10%) | 333/378 (88.10%) | 285/378 (75.40%) |
| RL 100 | 141/164 (85.98%) | 132/164 (80.49%) | 333/378 (88.10%) | 287/378 (75.93%) |
| RL 150 | 142/164 (86.59%) | 131/164 (79.88%) | 327/378 (86.51%) | 277/378 (73.28%) |
| RL 200 | 141/164 (85.98%) | 128/164 (78.05%) | 332/378 (87.83%) | 277/378 (73.28%) |

The checkpoints remain close to the base model on standard tests and regress on several plus metrics. The best checkpoint depends on the benchmark, and later is not consistently better.

### One-shot LiveCodeBench diverse50, official hidden tests

| Checkpoint | pass@1 |
| --- | ---: |
| Base | 15/50 (30.00%) |
| RL 50 | 14/50 (28.00%) |
| RL 100 | 13/50 (26.00%) |
| RL 150 | 13/50 (26.00%) |
| RL 200 | 15/50 (30.00%) |

This run used temperature 0.2. The RL checkpoints do not exceed the base result on this subset.

### RL checkpoint 200 with `react-bench`

| Benchmark | Runner/public repair result | Stricter post-hoc result |
| --- | ---: | ---: |
| HumanEval | 159/164 repair@3 (96.95%) | 160/164 base (97.56%); 141/164 plus (85.98%) |
| MBPP | 355/378 repair@3 (93.92%) | 355/378 base (93.92%); 294/378 plus (77.78%) |
| LCB diverse50, temp 0.6 | 41/50 public repair@3 (82.00%) | 36/50 hidden (72.00%) |
| LCB diverse50, temp 0.2 | 43/50 public repair@3 (86.00%) | 34/50 hidden (68.00%) |

Important qualifications:

- The HumanEval runner and official base count differ by one task because their harnesses differ.
- The MBPP+ audit contains 376 audited tasks; the reported 294/378 counts the two missing audits as failures.
- Public repair success is expected to be higher than hidden-test success because the agent acts on public feedback.
- Temperature 0.2 improves the public LCB repair number but reduces the corresponding hidden result from 72% to 68%, a useful example of feedback overfitting.

### What the comparison supports

Relative to RL checkpoint 200 one-shot evaluation, the repair system records the following changes:

| Metric | One-shot RL 200 | RL 200 + repair | Difference |
| --- | ---: | ---: | ---: |
| HumanEval+ | 78.05% | 85.98% | +7.93 percentage points |
| MBPP+ | 73.28% | 77.78% | +4.50 percentage points |
| LCB diverse50 hidden | 30.00% | 72.00% | +42.00 percentage points |

This is a **system-level** comparison, not a pure model comparison: the repair setup changes the interaction protocol, context budget, and number of generation opportunities.

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

1. official evaluator result JSON or hidden-test summary;
2. the machine-readable 8B `summary.json` recomputed from the archived copy;
3. `runs.jsonl` for runner-level behavior and audit reconstruction;
4. console logs and generated-only output files;
5. prose notes.

The 8B archive includes a [`MANIFEST.tsv`](../workstreams/model-post-training/results/qwen3_8b_coderl_react_eval_20260504/MANIFEST.tsv) with relative paths and byte sizes for provenance.

## Limitations

- The evaluation sets are small enough that a few tasks can move percentages materially.
- Some comparisons change sampling, context length, prompt format, or evaluation harness.
- Process-based execution reduces accidental interference but is not a hardened security sandbox for untrusted code.
- The archive does not include model weights, so stored generations and evaluator outputs are the primary evidence.
- A subset score should not be presented as a full benchmark score.
