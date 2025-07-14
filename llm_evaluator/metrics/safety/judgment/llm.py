from typing import Any

from ....inference import get_inference
from ....utils.type_utils import InferenceOutput
from .base import BaseAttackSuccessJudgment
from .prompts import get_prompt_builder
from .registry import AttackSuccessJudgmentRegistry


@AttackSuccessJudgmentRegistry.register("LLM")
class LlmAttackSuccessJudgment(BaseAttackSuccessJudgment):
    def __init__(self, judgment_cfgs: dict[str, Any]):
        super().__init__(judgment_cfgs=judgment_cfgs)
        self.model_cfgs: dict[str, Any] = self.judgment_cfgs.get("model_cfgs")  # type: ignore [assignment]
        self.inference_cfgs: dict[str, Any] = self.judgment_cfgs.get("inference_cfgs")  # type: ignore [assignment]
        prompt_builder_type = self.inference_cfgs.pop("prompt_builder_type")
        self.prompt_builder = get_prompt_builder(prompt_builder_type)
        self.inference = get_inference(
            model_cfgs=self.model_cfgs, inference_cfgs=self.inference_cfgs
        )

    def judge(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[float, dict[str, Any]]]:
        prompts = [self.prompt_builder.build_prompt(output) for output in outputs]
        judgments = self.inference.generate(
            prompts, enable_tqdm=True, tqdm_args={"desc": "Judging outputs"}
        )
        return [
            (self.prompt_builder.output2rate(judgment), judgment.model_dump())
            for judgment in judgments
        ]
