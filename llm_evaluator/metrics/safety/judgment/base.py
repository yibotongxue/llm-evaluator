from abc import ABC, abstractmethod
from typing import Any

from ....utils.shutdownable import Shutdownable
from ....utils.type_utils import InferenceOutput


class BaseAttackSuccessJudgment(ABC, Shutdownable):
    def __init__(self, judgment_cfgs: dict[str, Any]):
        self.judgment_cfgs = judgment_cfgs

    @abstractmethod
    def judge(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[bool, dict[str, Any]]]:
        pass
