from abc import ABC, abstractmethod

from ..utils.type_utils import InferenceInput, InferenceOutput


class BasePromptBuilder(ABC):
    @abstractmethod
    def process_input(self, raw_input: InferenceInput) -> InferenceInput:
        """
        Build a prompt from the raw prompt.

        Args:
            raw_prompt (InferenceInput): The raw prompt.

        Returns:
            InferenceInput: The built prompt.
        """

    @abstractmethod
    def parse_output(self, raw_output: InferenceOutput) -> InferenceOutput:
        """
        Parse the raw output to extract the answer.

        Args:
            raw_output (InferenceOutput): The raw output.

        Returns:
            InferenceOutput: The parsed output.
        """
