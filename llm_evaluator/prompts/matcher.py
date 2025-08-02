from abc import abstractmethod
from typing import override

from ..utils.extract_tools import extract_last_tag_content
from ..utils.type_utils import InferenceInput, InferenceOutput
from .base import BasePromptBuilder
from .registry import PromptBuilderRegistry


class MatcherPromptBuilder(BasePromptBuilder):
    @override
    def process_input(self, raw_input: InferenceInput) -> InferenceInput:
        ref_answer = raw_input.ref_answer
        if ref_answer is None:
            raise ValueError("Invalid sample: no reference answer found.")
        conversation = raw_input.conversation.copy()
        last_turn = conversation[-1].copy()
        answer: str = last_turn["content"]
        prompt = self._build_prompt(ref_answer, answer)
        last_turn["content"] = prompt
        conversation[-1] = last_turn
        raw = {
            **raw_input.model_dump(),
            "conversation": conversation,
        }
        return InferenceInput(**raw)

    @abstractmethod
    def _build_prompt(self, ref_answer: str, answer: str) -> str:
        pass


@PromptBuilderRegistry.register("MathMatch")
class MathMatchJudgmentPromptBuilder(MatcherPromptBuilder):
    @override
    def _build_prompt(self, ref_answer: str, answer: str) -> str:
        return f"""
You are a judge for a math problem. Your task is to determine whether the provided answer is correct or not according to the reference answer.
The provided answer and the reference answer may be a expression or a full solution. If they are both expressions, you should check if they are equivalent in the mathematical sense.
This means that you should check if they can be simplified to the same value and you should ignore other differences such as may be one of them contains a unit but the other does not.
The provided answer may also be a full solution, in this case you should check if the solution is correct and if it leads to the same value as the reference answer.
The reference answer is {ref_answer}
The provided answer is {answer}
You should first perform necessary analysis and then give your answer in the following format in the last line of your response:
<answer>Correct/Incorrect</answer>
The answer tag is necessary for automated evaluation. Fill "Correct" if the provided answer is correct, otherwise fill "Incorrect".
""".strip()

    @override
    def parse_output(self, raw_output: InferenceOutput) -> InferenceOutput:
        extracted_answer = extract_last_tag_content(raw_output.response, "answer")
        if extracted_answer is None:
            return raw_output
        is_correct = extracted_answer.strip().lower() == "correct"
        return raw_output.with_parsed_output(is_correct)
