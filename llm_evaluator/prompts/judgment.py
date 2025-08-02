from abc import abstractmethod
from typing import override

from ..utils.type_utils import InferenceInput, InferenceOutput
from .base import BasePromptBuilder
from .registry import PromptBuilderRegistry


class JudgmentPromptBuilder(BasePromptBuilder):
    @override
    def parse_output(self, raw_output: InferenceOutput) -> InferenceOutput:
        return raw_output.with_parsed_output(self._extract_answer(raw_output.response))

    @abstractmethod
    def _extract_answer(self, raw_output: str) -> bool:
        pass


@PromptBuilderRegistry.register("LlamaGuard")
class LlamaGuardPromptBuilder(JudgmentPromptBuilder):
    @override
    def process_input(self, raw_prompt: InferenceInput) -> InferenceInput:
        return raw_prompt

    @override
    def _extract_answer(self, raw_output: str) -> bool:
        return "unsafe" in raw_output.lower()
