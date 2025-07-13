from ..metrics import BaseMetricsComputer
from ..metrics.safety import get_safety_metrics
from .base import BaseBenchmark


class SafetyBenchmark(BaseBenchmark):
    def init_metrics(self) -> list[BaseMetricsComputer]:
        metrics_cfgs = self.eval_cfgs.metrics_cfgs
        return [get_safety_metrics(metrics_cfg) for metrics_cfg in metrics_cfgs]
