from typing import Any

from ...utils.type_utils import InferenceInput
from .base import BaseDataFormatter
from .registry import DataFormatterRegistry


class BaseSafetyDataFormatter(BaseDataFormatter):
    pass


@DataFormatterRegistry.register("AdvBench")
class AdvBenchDataFormatter(BaseSafetyDataFormatter):
    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        meta_data = {
            "prefill": raw_sample["target"],
            "raw_sample": raw_sample,
        }
        return InferenceInput.from_prompts(raw_sample["goal"]).with_meta_data(meta_data)


@DataFormatterRegistry.register("StrongReject")
class StrongRejectDataFormatter(BaseSafetyDataFormatter):
    def is_in_task_list(self, raw_sample: dict[str, Any], task_list: list[str]) -> bool:
        return raw_sample["category"] in task_list

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        return InferenceInput.from_prompts(
            raw_sample["forbidden_prompt"]
        ).with_meta_data({"meta_data": raw_sample})


@DataFormatterRegistry.register("JBB-Behaviors")
class JBBBehaviorsDataFormatter(BaseSafetyDataFormatter):
    def is_in_task_list(self, raw_sample: dict[str, Any], task_list: list[str]) -> bool:
        return raw_sample["Category"] in task_list

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        meta_data = {
            "prefill": raw_sample["Target"],
            "raw_sample": raw_sample,
        }
        return InferenceInput.from_prompts(raw_sample["Goal"]).with_meta_data(meta_data)
