from ..base import BasePromptBuilder
from ..chain import ChainPromptBuilder
from ..registry import PromptBuilderRegistry
from .base import AttackPromptBuilder


@PromptBuilderRegistry.register("ChainAttack")
class ChainAttackPromptBuilder(ChainPromptBuilder, AttackPromptBuilder):
    _prompt_builder_type: type[BasePromptBuilder] = AttackPromptBuilder  # type: ignore [type-abstract]
