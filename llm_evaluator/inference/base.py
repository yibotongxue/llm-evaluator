from abc import ABC, abstractmethod
from typing import Any

from ..utils.shutdownable import Shutdownable
from ..utils.type_utils import InferenceInput, InferenceOutput


class InferenceInterface(ABC):
    @abstractmethod
    def generate(
        self,
        inputs: list[InferenceInput],
        enable_tqdm: bool = False,
        tqdm_args: dict[str, Any] | None = None,
    ) -> list[InferenceOutput]:
        pass


class BaseInference(InferenceInterface, Shutdownable):
    def __init__(
        self, cfgs_hash: str, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        self._cfgs_hash = cfgs_hash
        self.model_cfgs = model_cfgs
        self.inference_cfgs = inference_cfgs

    @property
    def cfgs_hash(self) -> str:
        return self._cfgs_hash
