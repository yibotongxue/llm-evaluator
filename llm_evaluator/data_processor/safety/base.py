from abc import abstractmethod

from ...utils.type_utils import InferenceInput
from ..base import BaseDataProcessor


class BaseSafetyDataProcessor(BaseDataProcessor):
    @abstractmethod
    def format_conversation(self, raw_sample: dict[str, str]) -> InferenceInput:
        """
        生成对话。

        Args:
            raw_sample (dict[str, str]): 原始的样本输入数据。

        Returns:
            InferenceInput: 推理输入
        """

    @abstractmethod
    def format_prefilled_conversation(
        self, raw_sample: dict[str, str]
    ) -> InferenceInput:
        """
        生成预填充的对话。

        Args:
            raw_sample (dict[str, str]): 原始的样本输入数据。

        Returns:
            InferenceInput: 预填充的推理输入
        """
