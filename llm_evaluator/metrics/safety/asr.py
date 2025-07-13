from typing import Any

from ...utils.type_utils import InferenceOutput
from .base import BaseSafetyMetricsComputer
from .judgment import get_attack_success_judgment
from .registry import SafetyMetricsRegistry


@SafetyMetricsRegistry.register("ASR")
class ASRSafetyMetricsComputer(BaseSafetyMetricsComputer):
    def __init__(self, metrics_cfgs: dict[str, Any]):
        super().__init__(metrics_cfgs)
        self.judgment = get_attack_success_judgment(metrics_cfgs)

    def compute_metrics(self, outputs: list[InferenceOutput]) -> tuple[str, float]:
        judgments = self.judgment.judge(outputs)
        return self.metrics_name, sum(judgments) / len(judgments)
