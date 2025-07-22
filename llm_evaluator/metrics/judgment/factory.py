from typing import Any

from .base import BaseJudgment
from .registry import JudgmentRegistry


def get_judgment(judgment_cfgs: dict[str, Any]) -> BaseJudgment:
    """
    根据配置获取相应的判断器实例

    参数
    ----
    judgment_cfgs : dict[str, Any]
        判断器配置参数，必须包含'judgment_type'

    返回
    ----
    BaseJudgment
        相应的判断器实例
    """
    judgment_type = judgment_cfgs["judgment_type"]
    return JudgmentRegistry.get_by_name(judgment_type)(judgment_cfgs)
