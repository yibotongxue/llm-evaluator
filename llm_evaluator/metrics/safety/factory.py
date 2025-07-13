from typing import Any

from .base import BaseSafetyMetricsComputer
from .registry import SafetyMetricsRegistry


def get_safety_metrics(metrics_cfgs: dict[str, Any]) -> BaseSafetyMetricsComputer:
    metrics_type = metrics_cfgs.pop("metrics_type")
    return SafetyMetricsRegistry.get_by_name(metrics_type)(metrics_cfgs=metrics_cfgs)
