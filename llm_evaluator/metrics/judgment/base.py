from abc import ABC, abstractmethod
from typing import Any

from llm_evaluator.utils.type_utils import InferenceOutput


class BaseJudgment(ABC):
    def __init__(self, judgment_cfgs: dict[str, Any]):
        self.judgment_cfgs = judgment_cfgs

    @abstractmethod
    def judge(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[bool, dict[str, Any]]]:
        """对输出进行判断，返回一个元组列表，每个元组包含判断结果和相关信息"""
