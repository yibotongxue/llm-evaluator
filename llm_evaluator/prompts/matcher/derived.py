from ...utils.extract_tools import extract_last_tag_content
from .base import BaseMatcherJudgmentPromptBuilder
from .registry import MatcherPromptBuilderRegistry


@MatcherPromptBuilderRegistry.register("MATH")
class MathMatchJudgmentPromptBuilder(BaseMatcherJudgmentPromptBuilder):
    def build_prompt(self, ref_answer: str, answer: str) -> str:
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

    def extract_answer(self, raw_output: str) -> bool:
        extracted_answer = extract_last_tag_content(raw_output, "answer")
        if extracted_answer is None:
            return False
        return extracted_answer.strip().lower() == "correct"
