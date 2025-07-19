from abc import ABC, abstractmethod
from typing import Any

from ....utils.type_utils import InferenceOutput


class BaseCorrectJudgment(ABC):
    def __init__(self, judgment_cfgs: dict[str, Any]):
        self.judgment_cfgs = judgment_cfgs

    @abstractmethod
    def judge(self, outputs: list[InferenceOutput]) -> list[bool]:
        """Judge the correctness of the outputs.

        Args:
            outputs (list[InferenceOutput]): List of inference outputs to judge.

        Returns:
            list[bool]: List of boolean values indicating whether each output is correct.
        """
