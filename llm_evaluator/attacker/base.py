from abc import ABC, abstractmethod
from typing import Any

from ..utils.type_utils import InferenceInput


class BaseAttacker(ABC):
    def __init__(self, attack_cfgs: dict[str, Any]) -> None:
        self.attack_cfgs = attack_cfgs

    @abstractmethod
    def process_input(self, raw_input: InferenceInput) -> InferenceInput:
        """
        Process the raw input.

        Args:
            raw_input (InferenceInput): The raw input.

        Returns:
            InferenceInput: The processed input.
        """
