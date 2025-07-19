from typing import Any

from ...utils.type_utils import InferenceOutput, MetricsOutput
from .base import BaseCapabilityMetricsComputer
from .judgment import get_correct_judgment
from .registry import CapabilityMetricsRegistry


@CapabilityMetricsRegistry.register("Accuracy")
class AccuracyMetricsComputer(BaseCapabilityMetricsComputer):
    def __init__(self, metrics_cfgs: dict[str, Any]):
        super().__init__(metrics_cfgs)
        judgment_cfgs: dict[str, Any] = metrics_cfgs.get("judgment_cfgs")  # type: ignore [assignment]
        self.correct_judgment = get_correct_judgment(judgment_cfgs)

    def compute_metrics(self, outputs: list[InferenceOutput]) -> MetricsOutput:
        correctness = self.correct_judgment.judge(outputs)
        accuracy = sum(correctness) / len(correctness) if correctness else 0.0
        return MetricsOutput(
            metrics_name=self.metrics_name,
            metrics=accuracy,
            meta_data=[],
        )
