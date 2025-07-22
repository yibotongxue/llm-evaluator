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
