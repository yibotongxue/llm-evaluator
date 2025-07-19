from typing import Any

from ...prompts import BasePromptBuilder
from ...utils.type_utils import InferenceInput
from .base import BaseCapabilityDataFormatter
from .registry import CapabilityDataFormatterRegistry


@CapabilityDataFormatterRegistry.register("AIME")
class AIMECapabilityDataFormatter(BaseCapabilityDataFormatter):
    def __init__(self, prompt_builder: BasePromptBuilder):
        super().__init__(prompt_builder)

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        question = raw_sample["Problem"]
        prompt = self.prompt_builder.build_prompt(question)
        return InferenceInput(
            conversation=[{"role": "user", "content": prompt}],
            prefilled=False,
            system_prompt="",
            ref_answer=str(raw_sample["Answer"]),
            meta_data=raw_sample.copy(),
        )
