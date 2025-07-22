from typing import Any

from ..utils.type_utils import InferenceOutput, MetricsOutput
from .base import BaseMetricsComputer
from .judgment import JudgmentRegistry
from .registry import MetricsRegistry


@MetricsRegistry.register("Accuracy")
class AccuracyMetricsComputer(BaseMetricsComputer):
    def __init__(self, metrics_cfgs: dict[str, Any]):
        super().__init__(metrics_cfgs)
        judgment_cfgs: dict[str, Any] = metrics_cfgs.get("judgment_cfgs")  # type: ignore [assignment]
        judgment_type = judgment_cfgs["judgment_type"]
        self.judgment = JudgmentRegistry.get_by_name(judgment_type)(judgment_cfgs)

    def compute_metrics(self, outputs: list[list[InferenceOutput]]) -> MetricsOutput:
        for output_list in outputs:
            if not len(output_list) == 1:
                raise ValueError(
                    "Accuracy metrics expect outputs to be a list of lists with one output per input."
                )
        flatten_outputs = [output[0] for output in outputs]
        judgments = self.judgment.judge(flatten_outputs)
        correctness = [judgment[0] for judgment in judgments]
        judgment_meta_datas = [judgment[1] for judgment in judgments]
        accuracy = sum(correctness) / len(correctness) if correctness else 0.0
        return MetricsOutput(
            metrics_name=self.metrics_name,
            metrics=accuracy,
            meta_data={
                "raw_outputs": [output.response for output in flatten_outputs],
                "judgment_meta_datas": judgment_meta_datas,
            },
        )
