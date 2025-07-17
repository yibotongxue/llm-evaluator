from collections.abc import Callable
from typing import Any

from ..cache_manager import BaseCacheManager
from ..utils.logger import Logger
from ..utils.tools import dict_to_hash
from ..utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseInference, InferenceInterface


class CachedInference(InferenceInterface):
    def __init__(
        self, inference: BaseInference, cache_manager: BaseCacheManager
    ) -> None:
        self.inference = inference
        self.cache_manager = cache_manager
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    def generate(
        self,
        inputs: list[InferenceInput],
        enable_tqdm: bool = False,
        tqdm_args: dict[str, Any] | None = None,
    ) -> list[InferenceOutput]:
        cached_input_indices = []
        cached_result: list[InferenceOutput] = []
        for i, input in enumerate(inputs):
            input_key = self._generate_key(input)
            cache = self.cache_manager.load_cache(input_key)
            if cache is not None:
                cached_input_indices.append(i)
                cached_result.append(InferenceOutput(**cache))
        self.logger.info(
            f"一共有{len(inputs)}条请求，从缓存读取了{len(cached_input_indices)}条"
        )
        non_cached_inputs = [
            input for i, input in enumerate(inputs) if i not in cached_input_indices
        ]
        non_cached_outputs = self.inference.generate(
            non_cached_inputs, enable_tqdm, tqdm_args
        )
        for input, output in zip(non_cached_inputs, non_cached_outputs):
            key = self._generate_key(input)
            self.cache_manager.save_cache(key, output.model_dump())
        cached_idx = 0
        non_cached_idx = 0
        result: list[InferenceOutput] = []
        for i in range(len(inputs)):
            if i in cached_input_indices:
                result.append(cached_result[cached_idx])
                cached_idx += 1
            else:
                result.append(non_cached_outputs[non_cached_idx])
                non_cached_idx += 1
        return result

    def _update_inference_cfgs(
        self, new_inference_cfgs: dict[str, Any]
    ) -> Callable[[], None]:
        return self.inference._update_inference_cfgs(new_inference_cfgs)

    def _generate_key(self, inference_input: InferenceInput) -> str:
        key_message = {
            "system_prompt": inference_input.system_prompt,
            "conversation": inference_input.conversation,
            "cfgs_hash": self.inference.cfgs_hash,
            "prefilled": inference_input.prefilled,
        }
        return dict_to_hash(key_message)
