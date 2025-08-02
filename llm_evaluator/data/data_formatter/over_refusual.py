from typing import Any

from ...utils.type_utils import InferenceInput
from .base import BaseDataFormatter
from .registry import DataFormatterRegistry


class OverRefusualDataFormatter(BaseDataFormatter):
    pass


@DataFormatterRegistry.register("XSTest")
class XSTestDataFormatter(OverRefusualDataFormatter):
    def is_valid_sample(self, raw_sample: dict[str, Any]) -> bool:
        return (
            "prompt" in raw_sample
            and "type" in raw_sample
            and "label" in raw_sample
            and raw_sample["label"] == "safe"
        )

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        return InferenceInput.from_prompts(prompt=raw_sample["prompt"]).with_meta_data(
            {
                "raw_sample": raw_sample,
                "task": "XSTest",
            }
        )
