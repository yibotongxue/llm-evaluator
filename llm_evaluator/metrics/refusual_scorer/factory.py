from typing import Any

from .base import BaseRefusualScorer


def get_refusual_scorer(scorer_cfgs: dict[str, Any]) -> BaseRefusualScorer:
    scorer_type = scorer_cfgs["scorer_type"]
    if scorer_type == "llm":
        from .llm import LlmRefusualScorer

        return LlmRefusualScorer(scorer_cfgs)
    else:
        raise ValueError(f"评分器类型{scorer_type}不存在")
