from ..utils.registry import BaseRegistry
from .base import BasePromptBuilder


class PromptBuilderRegistry(BaseRegistry[BasePromptBuilder]):
    @classmethod
    def verify_type(cls, name: str, type_bound: type[BasePromptBuilder]) -> None:
        prompt_builder = cls.get_by_name(name)
        instance = prompt_builder()
        if not isinstance(instance, type_bound):
            raise ValueError(
                f"Prompt builder type '{name}' is not a valid {type_bound.__name__}."
            )
