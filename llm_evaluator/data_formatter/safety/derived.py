from ...utils.type_utils import InferenceInput
from .base import BaseSafetyDataFormatter
from .registry import SafetyDataFormatterRegistry


@SafetyDataFormatterRegistry.register("AdvBench")
class AdvBenchDataFormatter(BaseSafetyDataFormatter):
    def format_conversation(self, raw_sample: dict[str, str]) -> InferenceInput:
        return InferenceInput.from_prompts(raw_sample["prompt"]).with_meta_data(
            raw_sample
        )

    def format_prefilled_conversation(
        self, raw_sample: dict[str, str]
    ) -> InferenceInput:
        conversation = [
            {"role": "user", "content": raw_sample["prompt"]},
            {"role": "assistant", "content": raw_sample["target"]},
        ]
        return InferenceInput(
            conversation=conversation,
            prefilled=True,
            system_prompt="",
            meta_data=raw_sample,
        )
