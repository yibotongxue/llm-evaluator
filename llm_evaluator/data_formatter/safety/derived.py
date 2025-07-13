from ...utils.type_utils import InferenceInput
from .base import BaseSafetyDataFormatter
from .registry import SafetyDataFormatterRegistry


@SafetyDataFormatterRegistry.register("AdvBench")
class AdvBenchDataFormatter(BaseSafetyDataFormatter):
    def format_conversation(self, raw_sample: dict[str, str]) -> InferenceInput:
        meta_data = {
            "prefill": raw_sample["target"],
            "raw_sample": raw_sample,
        }
        return InferenceInput.from_prompts(raw_sample["prompt"]).with_meta_data(
            meta_data
        )
