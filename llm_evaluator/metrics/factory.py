from typing import Any

from .base import BaseMetricsComputer
from .registry import MetricsRegistry


def get_metrics_computer(metrics_cfgs: dict[str, Any]) -> BaseMetricsComputer:
    """
    根据配置获取相应的指标计算器实例

    参数
    ----
    metrics_cfgs : dict[str, Any]
        指标计算器配置参数，必须包含'metrics_type'

    返回
    ----
    BaseMetricsComputer
        相应的指标计算器实例
    """
    metrics_type = metrics_cfgs["metrics_type"]
    return MetricsRegistry.get_by_name(metrics_type)(metrics_cfgs)
