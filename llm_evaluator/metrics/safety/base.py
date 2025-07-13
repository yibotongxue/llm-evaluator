from typing import Any

from ..base import BaseMetricsComputer


class BaseSafetyMetricsComputer(BaseMetricsComputer):
    def __init__(self, metrics_cfgs: dict[str, Any]):
        super().__init__(metrics_cfgs)
