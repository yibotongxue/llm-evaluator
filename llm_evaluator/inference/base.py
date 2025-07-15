from abc import ABC, abstractmethod
from typing import Any

from ..utils.shutdownable import Shutdownable
from ..utils.type_utils import InferenceInput, InferenceOutput


class BaseInference(ABC, Shutdownable):
    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        self.model_cfgs = model_cfgs
        self.inference_cfgs = inference_cfgs

    @abstractmethod
    def generate(
        self,
        inputs: list[InferenceInput],
        enable_tqdm: bool = False,
        tqdm_args: dict[str, Any] | None = None,
    ) -> list[InferenceOutput]:
        pass
