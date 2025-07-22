from typing import Any

from ...utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseJudgment
from .registry import JudgmentRegistry


@JudgmentRegistry.register("ExactMatch")
class ExactMatchJudgment(BaseJudgment):
    def judge(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[bool, dict[str, Any]]]:
        for output in outputs:
            if InferenceInput(**output.input).ref_answer is None:
                raise ValueError("Input does not contain a reference answer.")

        results = [
            (
                output.extracted_answer is not None
                and output.extracted_answer.strip()
                == InferenceInput(**output.input).ref_answer.strip()  # type: ignore [union-attr]
            )
            for output in outputs
        ]

        return [(result, {}) for result in results]
