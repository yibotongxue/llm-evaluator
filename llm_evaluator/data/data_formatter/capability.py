import re
from typing import Any

from ...utils.type_utils import InferenceInput
from .base import BaseDataFormatter
from .registry import DataFormatterRegistry


class BaseCapabilityDataFormatter(BaseDataFormatter):
    """
    能力评估数据格式化器基类。

    专门用于处理与模型能力评估相关的数据格式化。
    """


@DataFormatterRegistry.register("AIME")
class AIMECapabilityDataFormatter(BaseCapabilityDataFormatter):
    """
    AIME能力评估数据格式化器。

    专门用于处理AIME（美国数学邀请赛）数据集的格式化。
    """

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        """
        格式化AIME对话数据。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据

        返回
        -------
        InferenceInput
            格式化后的推理输入
        """
        question = raw_sample["Problem"]
        return InferenceInput(
            conversation=[{"role": "user", "content": question}],
            prefilled=False,
            system_prompt="",
            ref_answer=str(raw_sample["Answer"]),
            meta_data=raw_sample.copy(),
        )


@DataFormatterRegistry.register("MATH")
class MathCapabilityDataFormatter(BaseCapabilityDataFormatter):
    """
    MATH Dataset数据格式化器。

    专门用于处理MATH Dataset的格式化。
    """

    def is_valid_sample(self, raw_sample: dict[str, Any]) -> bool:
        """
        我们直接尝试提取 \\boxed{} 中的答案，如果不能提取，则认为样本无效。

        TODO 或许需要更好的验证方法。
        """
        return self._get_ref_answer(raw_sample) is not None

    def is_in_task_list(self, raw_sample: dict[str, Any], task_list: list[str]) -> bool:
        """
        判断样本是否在任务列表中。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据
        task_list : list[str]
            任务列表

        返回
        -------
        bool
            如果样本类别在任务列表中则返回True，否则返回False
        """
        return raw_sample["type"] in task_list

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        """
        格式化MATH Dataset对话数据。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据

        返回
        -------
        InferenceInput
            格式化后的推理输入
        """
        question = raw_sample["problem"]
        ref_answer = self._get_ref_answer(raw_sample)
        if ref_answer is None:
            raise ValueError("Invalid sample: no reference answer found.")

        return InferenceInput(
            conversation=[{"role": "user", "content": question}],
            prefilled=False,
            system_prompt="",
            ref_answer=ref_answer,
            meta_data=raw_sample.copy(),
        )

    def _get_ref_answer(self, raw_sample: dict[str, Any]) -> str | None:
        """
        从原始样本中提取参考答案。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据

        返回
        -------
        str | None
            如果找到答案则返回，否则返回None
        """
        solution = raw_sample.get("solution")
        if solution is None or not isinstance(solution, str):
            return None

        boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        boxed_matches = re.findall(boxed_pattern, solution)
        if boxed_matches and len(boxed_matches) > 0:
            return boxed_matches[-1]  # type: ignore [no-any-return]
        return None


@DataFormatterRegistry.register("MMLUPro")
class MMLUProCapabilityDataFormatter(BaseCapabilityDataFormatter):
    """
    MMLU Pro能力评估数据格式化器。

    专门用于处理MMLU Pro数据集的格式化。
    """

    def is_valid_sample(self, raw_sample: dict[str, Any]) -> bool:
        """
        检查样本是否有效。

        MMLU Pro样本被认为是有效的，如果它：
        - 包含 "question"、"options" 和 "answer" 字段
        - "answer" 是一个单字符字符串（例如 'A'、'B'、'C' 等）
        - "options" 是一个列表，且其长度大于等于 "answer" 字符对应的选项索引
        """
        if (
            not "question" in raw_sample
            or not "options" in raw_sample
            or not "answer" in raw_sample
        ):
            return False
        ref_answer = raw_sample["answer"]
        if not isinstance(ref_answer, str) or len(ref_answer) != 1:
            return False
        if not isinstance(raw_sample["options"], list) or len(
            raw_sample["options"]
        ) <= ord(ref_answer) - ord("A"):
            return False
        return True

    def is_in_task_list(self, raw_sample: dict[str, Any], task_list: list[str]) -> bool:
        """
        判断样本是否在任务列表中。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据
        task_list : list[str]
            任务列表

        返回
        -------
        bool
            如果样本类别在任务列表中则返回True，否则返回False
        """
        return raw_sample["category"] in task_list

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        """
        格式化MMLU Pro对话数据。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据

        返回
        -------
        InferenceInput
            格式化后的推理输入
        """
        question = raw_sample["question"]
        options = raw_sample["options"]
        formatted_options = "\n".join(
            f"{chr(65 + i)}. {option}" for i, option in enumerate(options)
        )
        prompt = f"""
Question: {question}

Options:
{formatted_options}
"""
        return InferenceInput(
            conversation=[{"role": "user", "content": prompt}],
            prefilled=False,
            system_prompt="",
            ref_answer=raw_sample["answer"],
            meta_data=raw_sample.copy(),
        )
