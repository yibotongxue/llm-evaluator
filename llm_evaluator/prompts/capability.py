import re

from .base import BasePromptBuilder
from .registry import PromptBuilderRegistry


class CapabilityPromptBuilder(BasePromptBuilder):
    pass


# modified from https://github.com/UCSC-VLAA/STAR-1/blob/main/benchmark/reasoning_benchmark/aime_eval.py
@PromptBuilderRegistry.register("AIME")
class AIMEPromptBuilder(CapabilityPromptBuilder):
    def build_prompt(self, raw_prompt: str) -> str:
        return f"""
Solve the following AIME math problem step by step. The last line of your response should be of the form \\boxed{{X}} where X is the answer to the problem.

{raw_prompt}

IMPORTANT: Your final answer MUST be in the format \\boxed{{X}} where X is your numerical answer. This is critical for automated evaluation.
""".strip()

    def extract_answer(self, raw_output: str) -> str | None:
        BOXED_PATTERN = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        ANSWER_IS_PATTERN = r"(?i)(?:the\s+)?answer\s+(?:is|=)\s+(\d+)"
        LAST_LINE_NUMBER_PATTERN = r"(?m)^(\d+)$"

        # First try to find the last \boxed{} expression
        boxed_matches: list[str] = re.findall(BOXED_PATTERN, raw_output)

        if boxed_matches:
            # Get the content of the last \boxed{} expression
            raw_answer = boxed_matches[-1].strip()

            # Check if the answer is a simple number
            if raw_answer.isdigit():
                return raw_answer

            # For AIME problems, the answer is usually a number
            # If it's the last content in the boxed content, try to extract just the number
            number_match = re.search(r"(\d+)$", raw_answer)
            if number_match:
                return number_match.group(1)

            # If we can't extract a simple number, return the full content
            # This handles cases like "4\sqrt{7} - 3"
            print(f"Extracted complex answer from last \\boxed: {raw_answer}")
            return raw_answer

        # Fallback: Check for patterns like "The answer is 42" or "Answer: 42"
        answer_is_match = re.search(ANSWER_IS_PATTERN, raw_output)
        if answer_is_match:
            answer = answer_is_match.group(1).strip()
            print(f"Extracted answer from 'answer is' pattern: {answer}")
            return answer

        # Last resort: Check if the last line is just a number
        last_line_match = re.search(LAST_LINE_NUMBER_PATTERN, raw_output)
        if last_line_match:
            answer = last_line_match.group(1).strip()
            print(f"Extracted answer from last line: {answer}")
            return answer

        return None
