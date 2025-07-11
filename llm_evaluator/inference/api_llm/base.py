from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from ...utils.type_utils import InferenceInput, InferenctOutput
from ..base import BaseInference


class BaseApiLLMInference(BaseInference):
    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        super().__init__(model_cfgs=model_cfgs, inference_cfgs=inference_cfgs)
        self.max_retry: int = self.inference_cfgs.pop("max_retry", 3)

    def generate(self, inputs: list[InferenceInput]) -> list[InferenctOutput]:
        if len(inputs) == 1:
            return [self._single_generate(inputs[0])]
        return self._parallel_generate(inputs)

    def _parallel_generate(self, inputs: list[InferenceInput]) -> list[InferenctOutput]:
        for inference_input in inputs:
            if inference_input.prefilled:
                # TODO 需要设计更好的预填充方案
                last_messages = inference_input.conversation.pop(-1)
                if not last_messages["role"] == "assistant":
                    raise ValueError(
                        f"使用预填充的时候，最后一轮对话必须是assistant，但当前是{last_messages["role"]}"
                    )
                inference_input.conversation[-1]["content"] += last_messages["content"]
        results: dict[int, InferenctOutput] = {}
        max_workers = min(len(inputs), 32)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._single_generate, input_item): idx
                for idx, input_item in enumerate(inputs)
            }

            for future in tqdm(
                as_completed(future_to_index),
                total=len(inputs),
                desc="Generating responses",
            ):
                idx = future_to_index[future]
                result = future.result()
                results[idx] = result

        return [results[i] for i in range(len(inputs))]

    @abstractmethod
    def _single_generate(self, inference_input: InferenceInput) -> InferenctOutput:
        pass
