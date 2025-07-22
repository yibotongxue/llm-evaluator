from typing import Any

from ...utils.type_utils import InferenceInput
from .base import BaseDataFormatter
from .registry import DataFormatterRegistry


class BaseCapabilityDataFormatter(BaseDataFormatter):
    pass


@DataFormatterRegistry.register("AIME")
class AIMECapabilityDataFormatter(BaseCapabilityDataFormatter):

    def format_conversation(self, raw_sample: dict[str, Any]) -> InferenceInput:
        question = raw_sample["Problem"]
        return InferenceInput(
            conversation=[{"role": "user", "content": question}],
            prefilled=False,
            system_prompt="",
            ref_answer=str(raw_sample["Answer"]),
            meta_data=raw_sample.copy(),
        )
