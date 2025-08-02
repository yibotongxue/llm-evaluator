from typing import Any

from ...inference import InferenceFactory, InferenceInterface
from ...prompts import PromptBuilderRegistry
from ...prompts.over_refusual import OverRefusualPromptBuilder
from ...utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseRefusualScorer


class LlmRefusualScorer(BaseRefusualScorer):
    def __init__(self, scorer_cfgs: dict[str, Any]):
        super().__init__(scorer_cfgs)
        self.model_cfgs: dict[str, Any] = self.scorer_cfgs["model_cfgs"]
        self.inference_cfgs: dict[str, Any] = self.scorer_cfgs["inference_cfgs"]
        self.cache_cfgs: dict[str, Any] = self.scorer_cfgs.get("cache_cfgs", None)
        self.prompt_builder_type: str = self.scorer_cfgs["prompt_builder_type"]
        PromptBuilderRegistry.verify_type(
            self.prompt_builder_type, OverRefusualPromptBuilder  # type: ignore [type-abstract]
        )
        self.inference: InferenceInterface | None = None

    def score(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[float, dict[str, Any]]]:
        if self.inference is None:
            self.inference = InferenceFactory.get_inference_instance(
                model_cfgs=self.model_cfgs,
                inference_cfgs=self.inference_cfgs,
                cache_cfgs=self.cache_cfgs,
            )
        inputs = [
            InferenceInput.from_prompts(
                f"Question: {output.meta_data['raw_instruction']}\n\nAnswer: {output.response}"
            )
            for output in outputs
        ]
        judgments = self.inference.generate(
            inputs=inputs,
            repeat_cnt=1,
            prompt_template=self.prompt_builder_type,
            enable_tqdm=True,
            tqdm_args={"desc": "Judging outputs"},
        )
        results: list[tuple[float, dict[str, Any]]] = []
        for judgment in judgments:
            if not isinstance(judgment[0].parsed_output, float):
                raise ValueError(
                    f"Parsed output of judgment must be float, but got {judgment[0].parsed_output}."
                )
            results.append(
                (
                    judgment[0].parsed_output,
                    {"raw_output": judgment[0].response},
                )
            )
        return results
