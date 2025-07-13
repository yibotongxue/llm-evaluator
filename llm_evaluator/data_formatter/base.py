from abc import ABC, abstractmethod

from ..utils.type_utils import InferenceInput


class BaseDataFormatter(ABC):
    @abstractmethod
    def format_conversation(self, raw_sample: dict[str, str]) -> InferenceInput:
        """
        生成对话。

        Args:
            raw_sample (dict[str, str]): 原始的样本输入数据。

        Returns:
            InferenceInput: 推理输入
        """
