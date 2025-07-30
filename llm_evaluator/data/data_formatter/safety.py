from typing import Any

from ...utils.type_utils import InferenceInput
from .base import BaseDataFormatter
from .registry import DataFormatterRegistry


class BaseSafetyDataFormatter(BaseDataFormatter):
    """
    安全性评估数据格式化器基类。

    专门用于处理与模型安全性评估相关的数据格式化。
    """


@DataFormatterRegistry.register("AdvBench")
class AdvBenchDataFormatter(BaseSafetyDataFormatter):
    """
    AdvBench安全性评估数据格式化器。

    用于处理对抗基准（Adversarial Benchmark）数据集的格式化。
    """

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        """
        格式化AdvBench对话数据。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据

        返回
        -------
        InferenceInput
            格式化后的推理输入
        """
        meta_data = {
            "prefill": raw_sample["target"],
            "raw_sample": raw_sample,
        }
        return InferenceInput.from_prompts(raw_sample["goal"]).with_meta_data(meta_data)


@DataFormatterRegistry.register("StrongReject")
class StrongRejectDataFormatter(BaseSafetyDataFormatter):
    """
    强拒绝安全性评估数据格式化器。

    专门用于处理需要模型强拒绝的安全性评估数据集。
    """

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
        格式化强拒绝对话数据。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据

        返回
        -------
        InferenceInput
            格式化后的推理输入
        """
        return InferenceInput.from_prompts(
            raw_sample["forbidden_prompt"]
        ).with_meta_data({"meta_data": raw_sample})


@DataFormatterRegistry.register("JBB-Behaviors")
class JBBBehaviorsDataFormatter(BaseSafetyDataFormatter):
    """
    JBB行为安全性评估数据格式化器。

    专门用于处理JBB行为数据集的格式化。
    """

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
        return raw_sample["Category"] in task_list

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        """
        格式化JBB行为对话数据。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据

        返回
        -------
        InferenceInput
            格式化后的推理输入
        """
        meta_data = {
            "prefill": raw_sample["Target"],
            "raw_sample": raw_sample,
        }
        return InferenceInput.from_prompts(raw_sample["Goal"]).with_meta_data(meta_data)


@DataFormatterRegistry.register("wildjailbreak")
class WildJailbreakDataFormatter(BaseSafetyDataFormatter):
    """
    WildJailbreak安全性评估数据格式化器。

    用于处理WildJailbreak数据集的格式化。
    """

    def is_in_task_list(self, raw_sample: dict[str, Any], task_list: list[str]) -> bool:
        return raw_sample["data_type"] in task_list

    def is_valid_sample(self, raw_sample: dict[str, Any]) -> bool:
        if not "data_type" in raw_sample:
            return False
        data_type = raw_sample["data_type"]
        if data_type == "vanilla_harmful":
            return "vanilla" in raw_sample
        elif data_type == "adversarial_harmful":
            return "adversarial" in raw_sample
        else:
            return False

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        """
        格式化WildJailbreak对话数据。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据

        返回
        -------
        InferenceInput
            格式化后的推理输入
        """
        data_type = raw_sample["data_type"]
        if data_type == "vanilla_harmful":
            prompt_field = "vanilla"
        elif data_type == "adversarial_harmful":
            prompt_field = "adversarial"
        else:
            raise ValueError(
                f"Unknown supported data type for safety test: {data_type}"
            )
        return InferenceInput.from_prompts(raw_sample[prompt_field]).with_meta_data(
            {
                "raw_sample": raw_sample,
                "task": "wildjailbreak",
            }
        )
