from abc import ABC, abstractmethod

from .....utils.type_utils import InferenceInput, InferenceOutput


class BasePromptBuilder(ABC):
    @abstractmethod
    def build_prompt(self, output: InferenceOutput) -> InferenceInput:
        pass

    @abstractmethod
    def extract_from_output(self, output: InferenceOutput) -> bool:
        pass
