from typing import Any

from ....utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseCorrectJudgment
from .registry import CorrectJudgmentRegistry


@CorrectJudgmentRegistry.register("AIME")
class AIMECorrectJudgment(BaseCorrectJudgment):
    def __init__(self, judgment_cfgs: dict[str, Any]):
        super().__init__(judgment_cfgs)

    def judge(self, outputs: list[InferenceOutput]) -> list[bool]:
        for output in outputs:
            if output.extracted_answer is None:
                raise ValueError("Output does not contain an extracted answer.")
            if InferenceInput(**output.input).ref_answer is None:
                raise ValueError("Input does not contain a reference answer.")
        return [
            output.extracted_answer.strip()  # type: ignore [union-attr]
            == InferenceInput(**output.input).ref_answer.strip()  # type: ignore [union-attr]
            for output in outputs
        ]
