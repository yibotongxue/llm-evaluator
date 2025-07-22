from typing import Any

from ..utils.type_utils import InferenceOutput, InferSettings, MetricsOutput
from .base import BaseMetricsComputer
from .judgment import JudgmentRegistry
from .registry import MetricsRegistry


@MetricsRegistry.register("Pass@K")
class PassAtKMetricsComputer(BaseMetricsComputer):
    def __init__(self, metrics_cfgs: dict[str, Any]):
        super().__init__(metrics_cfgs)
        judgment_cfgs: dict[str, Any] = metrics_cfgs.get("judgment_cfgs")  # type: ignore [assignment]
        self.judgment = JudgmentRegistry.get_by_name(judgment_cfgs["type"])(
            judgment_cfgs
        )
        self.k = metrics_cfgs["k"]

    def infer_settings(self) -> InferSettings:
        return InferSettings(
            repeat_cnt=self.k,
            prompt_template=self.metrics_cfgs.get("prompt_template", None),
        )

    def compute_metrics(self, outputs: list[list[InferenceOutput]]) -> MetricsOutput:
        for output_list in outputs:
            if not len(output_list) == self.k:
                raise ValueError(
                    f"Pass@K metrics expect outputs to be a list of lists with {self.k} outputs per input."
                )
        flatten_outputs: list[InferenceOutput] = []
        for output in outputs:
            flatten_outputs.extend(output)
        judgments = self.judgment.judge(flatten_outputs)
        correctness = [judgment[0] for judgment in judgments]
        judgment_meta_datas = [judgment[1] for judgment in judgments]
        group_correctness = [
            sum(correctness[i : i + self.k]) >= 1
            for i in range(0, len(correctness), self.k)
        ]
        pass_at_k = sum(group_correctness) / len(group_correctness)
        return MetricsOutput(
            metrics_name=self.metrics_name,
            metrics=pass_at_k,
            meta_data={
                "raw_outputs": [output.response for output in flatten_outputs],
                "judgment_meta_datas": judgment_meta_datas,
            },
        )
