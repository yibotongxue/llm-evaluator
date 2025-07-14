from typing import Any

from ...utils.type_utils import InferenceOutput, MetricsOutput
from .base import BaseSafetyMetricsComputer
from .judgment import get_attack_success_judgment
from .registry import SafetyMetricsRegistry


@SafetyMetricsRegistry.register("ASR")
class ASRSafetyMetricsComputer(BaseSafetyMetricsComputer):
    def __init__(self, metrics_cfgs: dict[str, Any]):
        super().__init__(metrics_cfgs)
        judgment_cfgs: dict[str, Any] = metrics_cfgs.get("judgment_cfgs")  # type: ignore [assignment]
        self.judgment = get_attack_success_judgment(judgment_cfgs=judgment_cfgs)

    def compute_metrics(self, outputs: list[InferenceOutput]) -> MetricsOutput:
        judgments = self.judgment.judge(outputs)
        rates = [1.0 if judgment else 0.0 for judgment in judgments]
        meta_data = [judgment[1] for judgment in judgments]
        return MetricsOutput(
            metrics_name=self.metrics_name,
            metrics=sum(rates) / len(rates),
            meta_data=meta_data,
        )
