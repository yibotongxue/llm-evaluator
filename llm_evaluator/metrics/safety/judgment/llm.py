from typing import Any

from ....inference import BaseInference, InferenceFactory
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
        self.cache_cfgs: dict[str, Any] | None = self.judgment_cfgs.get(
            "cache_cfgs", None
        )
        prompt_builder_type = self.inference_cfgs.pop("prompt_builder_type")
        self.prompt_builder = get_prompt_builder(prompt_builder_type)
        self.inference: BaseInference | None = None

    def judge(
        self, outputs: list[InferenceOutput]
    ) -> list[tuple[bool, dict[str, Any]]]:
        if self.inference is None:
            self.inference = InferenceFactory.get_inference_instance(  # type: ignore [assignment]
                model_cfgs=self.model_cfgs,
                inference_cfgs=self.inference_cfgs,
                cache_cfgs=self.cache_cfgs,
            )
        prompts = [self.prompt_builder.build_prompt(output) for output in outputs]
        judgments = self.inference.generate(  # type: ignore [union-attr]
            prompts, enable_tqdm=True, tqdm_args={"desc": "Judging outputs"}
        )
        return [
            (self.prompt_builder.extract_from_output(judgment), judgment.model_dump())
            for judgment in judgments
        ]
