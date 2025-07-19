from typing import Any

from .base import BaseCapabilityMetricsComputer
from .registry import CapabilityMetricsRegistry


def get_capability_metrics(
    metrics_cfgs: dict[str, Any]
) -> BaseCapabilityMetricsComputer:
    metrics_type = metrics_cfgs.pop("metrics_type")
    return CapabilityMetricsRegistry.get_by_name(metrics_type)(
        metrics_cfgs=metrics_cfgs
    )
