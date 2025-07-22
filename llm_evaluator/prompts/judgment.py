from abc import abstractmethod

from .base import BasePromptBuilder
from .registry import PromptBuilderRegistry


class JudgmentPromptBuilder(BasePromptBuilder):
    def extract_answer(self, raw_output: str) -> str | None:
        return str(self._extract_answer(raw_output)).strip().lower()

    @abstractmethod
    def _extract_answer(self, raw_output: str) -> bool:
        pass


@PromptBuilderRegistry.register("LlamaGuard")
class LlamaGuardPromptBuilder(JudgmentPromptBuilder):
    def build_prompt(self, raw_prompt: str) -> str:
        return raw_prompt

    def _extract_answer(self, raw_output: str) -> bool:
        return "unsafe" in raw_output.lower()
