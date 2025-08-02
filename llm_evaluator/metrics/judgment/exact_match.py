from typing import Any

from ...utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseJudgment
from .registry import JudgmentRegistry


@JudgmentRegistry.register("ExactMatch")
class ExactMatchJudgment(BaseJudgment):
    def judge(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[bool, dict[str, Any]]]:
        ref_answers = [InferenceInput(**output.input).ref_answer for output in outputs]
        extracted_answers: list[str | None] = []
        for output in outputs:
            if output.parsed_output is not None and not isinstance(
                output.parsed_output, str
            ):
                raise ValueError(
                    f"Parsed output of judgment must be str or None, but got {output.parsed_output}."
                )
            extracted_answers.append(output.parsed_output)

        for ref_answer in ref_answers:
            if ref_answer is None:
                raise ValueError("Reference answer is None.")

        results = [
            extracted_answer is not None
            and ref_answer.strip() == extracted_answer.strip()  # type: ignore [union-attr]
            for ref_answer, extracted_answer in zip(ref_answers, extracted_answers)
        ]

        return [
            (result, {"ref_answer": ref_answer, "extracted_answer": extracted_answer})
            for result, (ref_answer, extracted_answer) in zip(
                results, zip(ref_answers, extracted_answers)
            )
        ]
