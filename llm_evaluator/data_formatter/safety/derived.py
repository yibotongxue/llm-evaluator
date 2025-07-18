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
        return InferenceInput.from_prompts(raw_sample["goal"]).with_meta_data(meta_data)


@SafetyDataFormatterRegistry.register("StrongReject")
class StrongRejectDataFormatter(BaseSafetyDataFormatter):
    def is_in_task_list(self, raw_sample: dict[str, str], task_list: list[str]) -> bool:
        return raw_sample["category"] in task_list

    def format_conversation(self, raw_sample: dict[str, str]) -> InferenceInput:
        return InferenceInput.from_prompts(
            raw_sample["forbidden_prompt"]
        ).with_meta_data({"meta_data": raw_sample})


@SafetyDataFormatterRegistry.register("JBB-Behaviors")
class JBBBehaviorsDataFormatter(BaseSafetyDataFormatter):
    def is_in_task_list(self, raw_sample: dict[str, str], task_list: list[str]) -> bool:
        return raw_sample["Category"] in task_list

    def format_conversation(self, raw_sample: dict[str, str]) -> InferenceInput:
        meta_data = {
            "prefill": raw_sample["Target"],
            "raw_sample": raw_sample,
        }
        return InferenceInput.from_prompts(raw_sample["Goal"]).with_meta_data(meta_data)
