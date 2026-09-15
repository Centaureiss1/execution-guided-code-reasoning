"""Custom reward functions for GRPO training (ms-swift ORM interface).

Registered in run_grpo.py via:
    from swift.rewards.orm import orms
    orms["code_exec"]   = CodeExecReward
    orms["code_format"] = CodeFormatReward

Dataset field used:
    tests: list[str]  — bare assert statements from grpo_train.jsonl
    Passed as kwargs["tests"] by ms-swift after batching dataset fields.
"""

import re
import subprocess
import sys
import textwrap
from typing import Optional

from swift.rewards.orm import ORM


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def extract_code(text: str) -> Optional[str]:
    """从 completion 中提取 Python 代码块。先去掉 </think> 前的思考内容。"""
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    # 优先提取 ```python ... ``` 块
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 退化：从第一个 def 开始取
    m2 = re.search(r"(def\s+\w+.*)", text, re.DOTALL)
    if m2:
        return textwrap.dedent(m2.group(0)).strip()
    return None


def run_tests(code: str, tests: list[str], timeout: int = 5) -> bool:
    """拼接 code + assert 语句，subprocess 执行，返回是否全部通过。"""
    import tempfile
    if not tests:
        return False
    full = code + "\n\n" + "\n".join(tests)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            r = subprocess.run(
                [sys.executable, "-c", full],
                timeout=timeout,
                capture_output=True,
                cwd=tmpdir,   # 隔离：代码创建的文件写到临时目录，自动清理
            )
        return r.returncode == 0
    except Exception:
        return False


# ── Reward 类 ─────────────────────────────────────────────────────────────────

class CodeExecReward(ORM):
    """主力 reward：代码通过所有 assert tests → 1.0，否则 0.0。

    GRPO 配置：reward_weights: [2.0, 0.5]（此 reward 权重 2.0）
    """

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        # ms-swift 将数据集字段 batched 后传入 kwargs
        # tests 字段来自 grpo_train.jsonl 的 "tests" key
        tests_batch = kwargs.get("tests", [[] for _ in completions])

        rewards = []
        for completion, tests in zip(completions, tests_batch):
            code = extract_code(completion)
            if code and run_tests(code, tests):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards


class CodeFormatReward(ORM):
    """辅助 reward：检查输出结构，防止模型退化成不写代码块。

    有 <think> 且有 ```python``` → 0.5
    只有 ```python```            → 0.3
    否则                          → 0.0

    GRPO 配置：reward_weights: [2.0, 0.5]（此 reward 权重 0.5）
    """

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for c in completions:
            has_think = "<think>" in c and "</think>" in c
            has_code  = "```python" in c
            if has_think and has_code:
                rewards.append(0.5)
            elif has_code:
                rewards.append(0.3)
            else:
                rewards.append(0.0)
        return rewards
