from typing import Any

from .base import BaseAttackSuccessJudgment
from .registry import AttackSuccessJudgmentRegistry


def get_attack_success_judgment(
    judgment_cfgs: dict[str, Any]
) -> BaseAttackSuccessJudgment:
    judgment_type = judgment_cfgs.pop("judgment_type")
    return AttackSuccessJudgmentRegistry.get_by_name(judgment_type)(
        judgment_cfgs=judgment_cfgs
    )
