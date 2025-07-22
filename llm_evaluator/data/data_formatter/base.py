from abc import ABC, abstractmethod
from typing import Any

from ...utils.type_utils import InferenceInput


class BaseDataFormatter(ABC):
    """
    数据格式化器基类。

    用于将原始数据样本转换为模型推理所需的标准格式。
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
            如果样本在任务列表中则返回True，否则返回False
        """
        return True

    @abstractmethod
    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        """
        格式化对话数据。

        参数
        ----------
        raw_sample : dict[str, Any]
            原始样本数据

        返回
        -------
        InferenceInput
            格式化后的推理输入
        """
