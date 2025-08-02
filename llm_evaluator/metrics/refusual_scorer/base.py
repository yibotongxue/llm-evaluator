from abc import ABC, abstractmethod
from typing import Any

from ...utils.type_utils import InferenceOutput


class BaseRefusualScorer(ABC):
    def __init__(self, scorer_cfgs: dict[str, Any]):
        self.scorer_cfgs = scorer_cfgs

    @abstractmethod
    def score(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[float, dict[str, Any]]]:
        pass
