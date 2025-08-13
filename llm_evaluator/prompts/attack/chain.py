from ..chain import ChainPromptBuilder
from ..registry import PromptBuilderRegistry
from .base import AttackPromptBuilder


@PromptBuilderRegistry.register("ChainAttack")
class ChainAttackPromptBuilder(ChainPromptBuilder, AttackPromptBuilder):
    pass
