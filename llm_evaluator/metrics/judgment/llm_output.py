from typing import Any

from ...inference import InferenceFactory, InferenceInterface
from ...prompts import PromptBuilderRegistry
from ...prompts.judgment import JudgmentPromptBuilder
from ...utils.type_utils import InferenceInput, InferenceOutput
from .base import BaseJudgment
from .registry import JudgmentRegistry


@JudgmentRegistry.register("LLM")
class LLMJudgment(BaseJudgment):
    def __init__(self, judgment_cfgs: dict[str, Any]):
        super().__init__(judgment_cfgs=judgment_cfgs)
        self.model_cfgs: dict[str, Any] = self.judgment_cfgs.get("model_cfgs")  # type: ignore [assignment]
        self.inference_cfgs: dict[str, Any] = self.judgment_cfgs.get("inference_cfgs")  # type: ignore [assignment]
        self.cache_cfgs: dict[str, Any] | None = self.judgment_cfgs.get(
            "cache_cfgs", None
        )
        self.prompt_builder_type = self.judgment_cfgs["prompt_builder_type"]
        PromptBuilderRegistry.verify_type(
            self.prompt_builder_type, JudgmentPromptBuilder  # type: ignore [type-abstract]
        )
        self.inference: InferenceInterface | None = None

    def judge(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[bool, dict[str, Any]]]:
        if self.inference is None:
            self.inference = InferenceFactory.get_inference_instance(
                model_cfgs=self.model_cfgs,
                inference_cfgs=self.inference_cfgs,
                cache_cfgs=self.cache_cfgs,
            )
        inputs = [
            InferenceInput.from_output(output, use_parsed_output=True)
            for output in outputs
        ]
        judgments = self.inference.generate(
            inputs=inputs,
            repeat_cnt=1,
            prompt_template=self.prompt_builder_type,
            enable_tqdm=True,
            tqdm_args={"desc": "Judging outputs"},
        )
        results: list[tuple[bool, dict[str, Any]]] = []
        for judgment in judgments:
            if not isinstance(judgment[0].parsed_output, bool):
                raise ValueError(
                    f"Parsed output of judgment must be bool, but got {judgment[0].parsed_output}."
                )
            results.append(
                (
                    judgment[0].parsed_output,
                    {"raw_output": judgment[0].response},
                )
            )
        return results
