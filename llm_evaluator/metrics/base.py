from abc import ABC, abstractmethod
from typing import Any

from ..utils.type_utils import InferenceOutput, InferSettings, MetricsOutput


class BaseMetricsComputer(ABC):
    def __init__(self, metrics_cfgs: dict[str, Any]):
        self.metrics_cfgs = metrics_cfgs
        self.metrics_name = self.metrics_cfgs.pop("metrics_name")

    def infer_settings(self) -> InferSettings:
        return InferSettings(
            repeat_cnt=1, prompt_template=self.metrics_cfgs.get("prompt_template", None)
        )

    @abstractmethod
    def compute_metrics(self, outputs: list[list[InferenceOutput]]) -> MetricsOutput:
        pass
