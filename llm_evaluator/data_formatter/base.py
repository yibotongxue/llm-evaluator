from abc import ABC, abstractmethod

from ..utils.type_utils import InferenceInput


class BaseDataFormatter(ABC):
    def is_in_task_list(
        self, raw_sample: dict[str, str], task_list: list[str] | None
    ) -> bool:
        if task_list is None:
            return True
        return self._is_in_task_list(raw_sample, task_list)

    def _is_in_task_list(
        self, raw_sample: dict[str, str], task_list: list[str]
    ) -> bool:
        return True

    @abstractmethod
    def format_conversation(self, raw_sample: dict[str, str]) -> InferenceInput:
        """
        生成对话。

        Args:
            raw_sample (dict[str, str]): 原始的样本输入数据。

        Returns:
            InferenceInput: 推理输入
        """
