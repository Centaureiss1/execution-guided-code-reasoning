# CS639 Final Project 仓库说明

[English Version](README.md)

## 项目概览

这个仓库围绕 `Qwen3-1.7B` 做代码生成实验，主线 workflow 是：

```text
数据准备 -> SFT -> merge 后的 SFT 模型 -> 构建 GRPO pool -> 离线打分与分档
-> GRPO 训练 -> HumanEval/MBPP 评测 -> 结果快照
```

除了 Qwen3 主线之外，仓库里还包含本地 baseline 的评测流程，主要对应 `Qwen2.5-Coder-1.5B-Instruct` 和 `DeepSeek-R1-Distill-Qwen-1.5B`，以及一些保留下来的历史评测结果。

## 仓库结构总览

| 路径 | 作用 |
| --- | --- |
| `scripts/data/` | 数据下载、过滤、去污染、构建候选池、离线打分 |
| `scripts/train/1.7B/` | SFT 启动脚本、GRPO 启动脚本、reward 定义、YAML 配置 |
| `scripts/eval/` | 采样脚本和 EvalPlus 封装 |
| `data/processed/` | 衍生出来的 JSONL 数据集和 GRPO 训练变体 |
| `models/` | 本地模型权重、LoRA checkpoint、merge 后模型、GRPO 训练产物 |
| `results/` | 当前主评测脚本产出的正式结果目录 |
| `scripts/eval_results/` | 放在 `scripts/` 下面的早期或临时评测结果，不是可执行脚本 |
| `logs/` | 实验日志目录 |
| `CS639 Proposal.pdf` | 项目 proposal |

## Stage 0：环境与依赖

当前脚本默认依赖一个本地 Python 环境，至少需要这些包：

- `datasets`
- `transformers`
- `vllm`
- `evalplus`
- `ms-swift`

这个仓库还有几个很重要的环境假设：

- 大多数脚本把仓库根目录写死为 `/home/brui/cs639_final`
- 训练和推理默认都是本地 GPU + `bfloat16`
- Qwen3 的训练配置使用 `attn_impl: sdpa`，没有走 FlashAttention
- 本地模型目录默认放在 `models/` 下面，尤其是：
  - `models/Qwen3-1.7B`
  - `models/Qwen3-1.7B/phase1_sft_merged`
  - `models/Qwen2.5-Coder-1.5B-Instruct`
  - `models/DeepSeek-R1-Distill-Qwen-1.5B`

## Pipeline 分阶段说明

### Stage 1：数据摸底脚本

**脚本：** `scripts/data/download_data.py`

这个脚本只是一个小工具，不是主流程必跑步骤。它会把 `ise-uiuc/Magicoder-OSS-Instruct-75K` 下载到 `data/raw/`，然后打印字段名和第一条样本，方便你在写预处理前先确认数据 schema。

典型用法：

```bash
python scripts/data/download_data.py
```

输入和输出：

- 输入：Hugging Face 上的 `Magicoder-OSS-Instruct-75K`
- 输出：缓存到 `data/raw/` 的原始数据
- 是否主流程必需：否

### Stage 2：构建 SFT 训练集

**脚本：** `scripts/data/prepare_sft.py`

这个脚本负责生成 `data/processed/sft_train.jsonl`，也就是 Phase 1 SFT 用的 chat 格式训练集。它会把两个上游数据源合并起来：

- `nvidia/OpenCodeInstruct`
- `ise-uiuc/Magicoder-OSS-Instruct-75K`

它具体做的事情包括：

- 对 OpenCodeInstruct 做 `average_test_score >= 0.9` 过滤
- 对 Magicoder 只保留 Python 样本
- 去掉空输入和空输出
- 去掉过长样本和过短答案
- 按 prompt 文本做去重
- 按固定配比采样：
  - 50,000 条 OpenCodeInstruct
  - 20,000 条 Magicoder
- 用固定随机种子打乱合并结果

当前输出：

- `data/processed/sft_train.jsonl`
- 这个仓库快照里大约有 **70,000** 条
- 记录格式是：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

典型用法：

```bash
python scripts/data/prepare_sft.py
```

### Stage 3：运行 SFT，然后把 LoRA merge 成完整模型

**启动脚本：** `scripts/train/1.7B/run_sft.sh`  
**配置文件：** `scripts/train/1.7B/sft_config.yaml`

这一阶段会在 `models/Qwen3-1.7B` 基础上做 LoRA SFT，并把 checkpoint 写到 `models/Qwen3-1.7B/phase1_sft/`。

几个关键配置点：

- 基座模型：`models/Qwen3-1.7B`
- 数据集：`data/processed/sft_train.jsonl`
- 训练类型：LoRA
- `loss_scale: ignore_empty_think`
- 输出目录：`models/Qwen3-1.7B/phase1_sft`

典型用法：

```bash
bash scripts/train/1.7B/run_sft.sh
```

这里有一个很关键的衔接点：

- 后面的步骤**不会**直接吃 `phase1_sft/` 里的 LoRA checkpoint 目录
- `score_and_tier.py` 和默认的 SFT 初始化 GRPO 配置，实际都假设你已经有一个 merge 后的完整模型：`models/Qwen3-1.7B/phase1_sft_merged`
- 仓库里没有单独的 Phase 1 merge 脚本，所以这一步是一个手动桥接步骤

可以参考 `scripts/eval/run_eval.py` 里的 merge 逻辑，手动执行类似命令：

```bash
python -m swift.cli.merge_lora \
  --model models/Qwen3-1.7B \
  --adapters models/Qwen3-1.7B/phase1_sft/<run>/checkpoint-<step> \
  --output_dir models/Qwen3-1.7B/phase1_sft_merged \
  --torch_dtype bfloat16
```

### Stage 4：构建 GRPO 候选池

**脚本：** `scripts/data/build_grpo_pool.py`

这个脚本会把多个来源的数据统一整理成一个适合 GRPO 的候选池 JSONL，是从监督数据过渡到 RL 风格训练数据的关键一步。

上游来源：

- `OpenCodeInstruct`
- `KodCode/KodCode-Light-RL-10K`

它会做的事情包括：

- 从 OpenCodeInstruct 中抽高质量题目
- 把 OpenCodeInstruct 的 `unit_tests` JSON 字符串转成 bare `assert` 列表
- 把 KodCode 的 pytest 风格测试转成 bare `assert`
- 去掉和以下集合重叠的题目：
  - HumanEval
  - MBPP 全部 split
  - SFT 训练集
- 在 OpenCodeInstruct 和 KodCode 之间做跨数据集去重
- 给每道题包一层统一的 system + user chat prompt

当前输出：

- `data/processed/grpo_pool.jsonl`
- 这个仓库快照里大约有 **49,825** 条
- 记录格式是：

```json
{"messages": [...], "tests": ["assert ..."], "source": "...", "question_id": "..."}
```

典型用法：

```bash
python scripts/data/build_grpo_pool.py
```

### Stage 5：离线打分与分档

**脚本：** `scripts/data/score_and_tier.py`

这个脚本会用 merge 后的 SFT 模型给每道 GRPO 候选题做离线打分，再把题目筛成适合 GRPO 训练的数据集。

它的流程是：

- 读取 `data/processed/grpo_pool.jsonl`
- 用 `models/Qwen3-1.7B/phase1_sft_merged` 做 vLLM 批量生成
- 每道题生成多个 completion
- 从模型输出里抽取 Python 代码块
- 把代码和题目的 `assert` 测试拼起来，在隔离子进程里执行
- 计算每道题的 `pass_rate`
- 把可断点续跑的中间结果写到 `grpo_pool_scored.jsonl`
- 最后根据打分生成 `grpo_train.jsonl`

当前输出：

- `data/processed/grpo_pool_scored.jsonl`，大约 **49,825** 条
- `data/processed/grpo_train.jsonl`，大约 **10,283** 条

支持的 CLI 选项：

- `--score-only`：只打分，不生成最终训练集
- `--tier-only`：跳过推理，只根据已有 scored 文件重建分档结果
- `--n-samples`：调整每道题采样多少个 completion

典型用法：

```bash
python scripts/data/score_and_tier.py
python scripts/data/score_and_tier.py --score-only
python scripts/data/score_and_tier.py --tier-only
python scripts/data/score_and_tier.py --n-samples 4
```

### Stage 5.5：实验性 GRPO 数据变体

`data/processed/` 下面目前还有几份额外的 GRPO 相关数据集，但生成这些特定变体的脚本并没有一起保存在当前仓库里：

| 文件 | 当前大小 | 大致含义 |
| --- | --- | --- |
| `grpo_train.jsonl` | 10,283 | `score_and_tier.py` 产出的默认分档训练集 |
| `grpo_train_curriculum.jsonl` | 10,283 | curriculum 风格实验变体 |
| `grpo_train_shuffle.jsonl` | 10,283 | shuffle 风格实验变体 |
| `grpo_train_hard.jsonl` | 5,018 | 当前 GRPO 配置正在使用的 hard 子集 |
| `grpo_mbpp_train.jsonl` | 374 | 单独整理出来的 MBPP 风格 GRPO 数据 |

这里有两个很容易混淆的点：

- `scripts/train/1.7B/grpo_config.yaml` 当前指向的是 `grpo_train_hard.jsonl`，不是 `grpo_train.jsonl`
- 如果你想完整复现这些变体数据的生成过程，当前仓库里还缺对应的生成脚本

### Stage 6：GRPO 训练

**启动脚本：** `scripts/train/1.7B/run_grpo.py`  
**Reward 定义：** `scripts/train/1.7B/reward_fn.py`  
**配置文件：**

- `scripts/train/1.7B/grpo_config.yaml`
- `scripts/train/1.7B/grpo_config_base_hard.yaml`

这一阶段通过 `ms-swift` 跑 GRPO，但和直接命令行调用 `swift rlhf` 不一样。原因是 reward 是本地自定义的 Python 类，所以训练必须通过 `run_grpo.py` 启动，在进程里先把 reward 注册进去。

`reward_fn.py` 里定义了两个 reward：

- `CodeExecReward`：抽取 Python 代码并执行测试，全部通过才给 `1.0`
- `CodeFormatReward`：鼓励模型保留预期的思考和代码块输出格式

`run_grpo.py` 做的事情是：

- 把当前训练目录加入 `sys.path`
- 把自定义 reward 注册到 `swift.rewards.orm.orms`
- 读取 YAML 配置
- 允许通过命令行覆盖 `max_steps`
- 最终调用 `swift.pipelines.train.rlhf.rlhf_main(...)`

两个配置文件的分工：

- `grpo_config.yaml`
  - 从 `models/Qwen3-1.7B/phase1_sft_merged` 出发
  - 当前使用 `data/processed/grpo_train_hard.jsonl`
  - 输出到 `models/Qwen3-1.7B/phase2_grpo_hard`
- `grpo_config_base_hard.yaml`
  - 从 base `models/Qwen3-1.7B` 出发
  - 同样使用 `data/processed/grpo_train_hard.jsonl`
  - 输出到 `models/Qwen3-1.7B/phase2_grpo_base_hard`

支持的 CLI 选项：

- `--steps`：覆盖 YAML 里的 `max_steps`
- `--config`：切换 GRPO 配置文件

典型用法：

```bash
python scripts/train/1.7B/run_grpo.py
python scripts/train/1.7B/run_grpo.py --steps 10
python scripts/train/1.7B/run_grpo.py --config scripts/train/1.7B/grpo_config_base_hard.yaml --steps 800
```

### Stage 7：评测主线 Qwen3 模型

**脚本：**

- `scripts/eval/gen_samples.py`
- `scripts/eval/run_eval.py`

这是主线 Qwen3 模型的评测路径，覆盖：

- base `Qwen3-1.7B`
- merge 后的 SFT 模型
- GRPO 的 LoRA checkpoint

它的工作方式是：

- `run_eval.py` 先判断 `--model` 指向的是完整模型还是 LoRA adapter checkpoint
- 如果是 adapter，就先自动 merge 到 `results/<exp>/merged_model`
- 然后调用 `gen_samples.py` 为 HumanEval 和 MBPP 生成解答
- 最后调用 `python -m evalplus.evaluate` 做评测

`gen_samples.py` 主要负责：

- 按 Qwen chat template 组织 prompt
- 处理 `thinking`、`auto`、`no_thinking` 三种模式
- 从原始输出里挑选最合适的代码块
- 生成 EvalPlus 兼容的 JSONL 样本文件

`run_eval.py` 支持的 CLI 选项：

- `--model`
- `--base-model`
- `--output-dir`
- `--thinking`
- `--auto`
- `--skip-gen`
- `--max-new-tokens`

典型用法：

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

### Stage 8：评测本地 baseline 模型

**脚本：**

- `scripts/eval/gen_samples_vllm_generic.py`
- `scripts/eval/run_eval_vllm_generic.py`

这套脚本用于本地非 Qwen3 模型的 baseline 评测。在当前仓库里，主要就是：

- `Qwen2.5-Coder-1.5B-Instruct`
- `DeepSeek-R1-Distill-Qwen-1.5B`

它和主线 Qwen3 评测的区别主要在于：

- 它会根据不同 family 使用不同的 prompt 组织方式和采样默认值
- 它的输出模式是 `native`，而不是 Qwen3 那套 `thinking/auto/no_thinking`
- 它面向的是完整本地模型，不是 LoRA adapter

`run_eval_vllm_generic.py` 支持的 CLI 选项：

- `--family`
- `--dataset`
- `--max-model-len`
- `--max-new-tokens`
- `--limit`
- `--skip-gen`

典型用法：

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

### Stage 9：结果目录说明

`results/` 目录存放的是当前主评测脚本生成的正式结果。在这个仓库快照里，比较关键的子目录有：

| 目录 | 含义 |
| --- | --- |
| `results/base_v2/` | base `Qwen3-1.7B` 的评测结果快照 |
| `results/sft_v2/` | merge 后 SFT 模型的评测结果快照 |
| `results/grpo_hard_800/` | 从 SFT 出发、在 hard 数据集上做 GRPO、checkpoint 800 的评测结果 |
| `results/grpo_base_hard_800/` | 从 base 模型出发、在 hard 数据集上做 GRPO、checkpoint 800 的评测结果 |
| `results/grpo_shuffle_600/` | shuffle 变体 GRPO 在 checkpoint 600 的评测结果 |
| `results/qwen25_coder_1_5b_instruct/` | 本地 Qwen2.5-Coder baseline 的评测结果 |
| `results/deepseek_r1_distill_qwen_1_5b_auto/` | 本地 DeepSeek-R1-Distill baseline 的评测结果 |

这里还有两个容易忽略的细节：

- `results/*/samples/...` 下面放的是生成的 JSONL 样本和 EvalPlus 结果 JSON
- `results/*/merged_model/` 可能会被 `run_eval.py` 自动创建，用来临时存放评测前 merge 出来的完整模型

历史说明：

- `scripts/eval_results/*` 里放的是保存在 `scripts/` 目录下的旧评测结果
- 它们更像归档参考，不是当前推荐的评测入口

## 附录 A：脚本索引表

| 脚本 | 主要作用 | 主要输入 | 主要输出 | 是否主线必需 | 典型运行时机 |
| --- | --- | --- | --- | --- | --- |
| `scripts/data/download_data.py` | 快速查看 Magicoder 数据 schema | Hugging Face 数据集 | 原始缓存数据 | 否 | 一次性摸底 |
| `scripts/data/prepare_sft.py` | 构建 SFT 训练集 | OpenCodeInstruct、Magicoder | `sft_train.jsonl` | 是 | Phase 1 SFT 之前 |
| `scripts/train/1.7B/run_sft.sh` | 启动 SFT | `sft_config.yaml`、Qwen3 base model | `phase1_sft/` 下的 LoRA checkpoint | 是 | Phase 1 训练 |
| `scripts/data/build_grpo_pool.py` | 构建去污染后的 GRPO 候选池 | OCI、KodCode、评测集指纹、SFT prompt | `grpo_pool.jsonl` | 是 | SFT 数据整理后 |
| `scripts/data/score_and_tier.py` | 用 merge 后 SFT 模型打分并生成 GRPO 训练集 | `grpo_pool.jsonl`、`phase1_sft_merged` | `grpo_pool_scored.jsonl`、`grpo_train.jsonl` | 是 | GRPO 前 |
| `scripts/train/1.7B/reward_fn.py` | 定义自定义 GRPO reward | 模型 completion、数据集里的 `tests` | 进程内 reward 分数 | 间接必需 | GRPO 启动时被导入 |
| `scripts/train/1.7B/run_grpo.py` | 注册 reward 并启动 GRPO | GRPO YAML 配置、reward 类 | GRPO checkpoint 和日志 | 是 | Phase 2 训练 |
| `scripts/eval/gen_samples.py` | 生成主线 Qwen3 评测样本 | 主模型路径、HumanEval/MBPP | 样本 JSONL | 是 | 由 `run_eval.py` 调用 |
| `scripts/eval/run_eval.py` | 组织 Qwen3 评测和 LoRA merge | 模型路径、可选 base model | `results/*` 评测结果 | 是 | 训练结束后 |
| `scripts/eval/gen_samples_vllm_generic.py` | 生成 baseline 模型评测样本 | baseline 模型路径、family | 样本 JSONL | 只对 baseline 必需 | baseline 评测 |
| `scripts/eval/run_eval_vllm_generic.py` | 组织 baseline 评测 | baseline 模型路径、family | `results/*` baseline 结果 | 只对 baseline 必需 | baseline 评测 |

### 关键配置文件

这些不是脚本，但理解 pipeline 时很重要：

| 文件 | 作用 |
| --- | --- |
| `scripts/train/1.7B/sft_config.yaml` | Phase 1 SFT 配置 |
| `scripts/train/1.7B/grpo_config.yaml` | 以 SFT 模型初始化、使用 hard 数据集的 GRPO 配置 |
| `scripts/train/1.7B/grpo_config_base_hard.yaml` | 以 base 模型初始化、使用 hard 数据集的 GRPO 配置 |

## 附录 B：常见坑

- **Phase 1 结束后需要先 merge。** 下游打分脚本和默认的 SFT 初始化 GRPO 配置都要求 `models/Qwen3-1.7B/phase1_sft_merged`，不能直接给它一个纯 LoRA checkpoint 目录。
- **不要把 `run_grpo.py` 直接替换成 `swift rlhf`。** 因为这个 Python 启动脚本会在训练前把自定义 reward 注入 `ms-swift` 的注册表；少了这一步，配置里的 reward 名称就解析不到。
- **只有评测 LoRA adapter checkpoint 时才需要 `--base-model`。** 如果 `--model` 本身已经是 merge 后的完整模型，`run_eval.py` 可以直接评测。
- **当前 GRPO 配置默认吃的是 `grpo_train_hard.jsonl`。** 这和 `score_and_tier.py` 默认产出的 `grpo_train.jsonl` 不是同一个文件。
- **很多路径是绝对路径。** 如果你把仓库挪位置，很多脚本需要先改路径才能正常跑。
- **`scripts/eval_results/*` 是历史结果，不是当前代码入口。** 当前推荐的评测入口还是 `scripts/eval/*.py`。
