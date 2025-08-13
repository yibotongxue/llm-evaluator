from typing import Any

from ..utils.type_utils import InferenceInput, InferenceOutput
from .base import BasePromptBuilder
from .registry import PromptBuilderRegistry


@PromptBuilderRegistry.register("chain")
class ChainPromptBuilder(BasePromptBuilder):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        prompt_cfg_list = config["prompt_cfg_list"]
        self.prompt_builder_list: list[BasePromptBuilder] = [
            PromptBuilderRegistry.get_by_name(prompt_cfg["type"])(prompt_cfg["config"])
            for prompt_cfg in prompt_cfg_list
        ]

    def process_input_list(
        self, raw_inputs: list[InferenceInput]
    ) -> list[InferenceInput]:
        for prompt_builder in self.prompt_builder_list:
            raw_inputs = prompt_builder.process_input_list(raw_inputs)
        return raw_inputs

    def parse_output_list(
        self, raw_outputs: list[InferenceOutput]
    ) -> list[InferenceOutput]:
        for prompt_builder in self.prompt_builder_list[::-1]:
            raw_outputs = prompt_builder.parse_output_list(raw_outputs)
        return raw_outputs

    def process_input(self, raw_input: InferenceInput) -> InferenceInput:
        for prompt_builder in self.prompt_builder_list:
            raw_input = prompt_builder.process_input(raw_input)
        return raw_input

    def parse_output(self, raw_output: InferenceOutput) -> InferenceOutput:
        for prompt_builder in self.prompt_builder_list[::-1]:
            raw_output = prompt_builder.parse_output(raw_output)
        return raw_output
