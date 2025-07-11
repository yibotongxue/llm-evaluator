from abc import ABC, abstractmethod
from typing import Any

from ..utils.type_utils import InferenceInput, InferenctOutput


class BaseInference(ABC):
    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        self.model_cfgs = model_cfgs
        self.inference_cfgs = inference_cfgs

    @abstractmethod
    def generate(self, inputs: list[InferenceInput]) -> list[InferenctOutput]:
        pass
