from typing import Any

from .base import BaseCorrectJudgment
from .registry import CorrectJudgmentRegistry


def get_correct_judgment(judgment_cfgs: dict[str, Any]) -> BaseCorrectJudgment:
    judgment_type = judgment_cfgs["judgment_type"]
    return CorrectJudgmentRegistry.get_by_name(judgment_type)(judgment_cfgs)
