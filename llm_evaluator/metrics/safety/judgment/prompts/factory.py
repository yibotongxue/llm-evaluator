from .base import BasePromptBuilder
from .registry import PromptBuilderRegistry


def get_prompt_builder(prompt_builder_type: str) -> BasePromptBuilder:
    return PromptBuilderRegistry.get_by_name(prompt_builder_type)()
