from abc import ABC, abstractmethod
from typing import Any

from ..utils.type_utils import InferenceOutput


class BaseMetricsComputer(ABC):
    def __init__(self, metrics_cfgs: dict[str, Any]):
        self.metrics_cfgs = metrics_cfgs
        self.metrics_name = self.metrics_cfgs.pop("metrics_name")

    @abstractmethod
    def compute_metrics(self, outputs: list[InferenceOutput]) -> tuple[str, float]:
        pass
