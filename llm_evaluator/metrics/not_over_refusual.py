from typing import Any

from ..utils.type_utils import InferenceOutput, MetricsOutput
from .base import BaseMetricsComputer
from .refusual_scorer import get_refusual_scorer
from .registry import MetricsRegistry


@MetricsRegistry.register("NotOverRefusual")
class NotOverRefusualMetricsComputer(BaseMetricsComputer):
    def __init__(self, metrics_cfgs: dict[str, Any]):
        super().__init__(metrics_cfgs)
        scorer_cfgs: dict[str, Any] = metrics_cfgs.get("scorer_cfgs")  # type: ignore [assignment]
        self.scorer = get_refusual_scorer(scorer_cfgs)

    def compute_metrics(self, outputs: list[list[InferenceOutput]]) -> MetricsOutput:
        for output_list in outputs:
            if not len(output_list) == 1:
                raise ValueError(
                    "Accuracy metrics expect outputs to be a list of lists with one output per input."
                )
        flatten_outputs = [output[0] for output in outputs]
        judgments = self.scorer.score(flatten_outputs)
        compilance_scores = [judgment[0] for judgment in judgments]
        judgment_meta_datas = [judgment[1] for judgment in judgments]
        not_over_refusual = (
            sum(compilance_scores) / len(compilance_scores)
            if compilance_scores
            else 0.0
        )
        return MetricsOutput(
            metrics_name=self.metrics_name,
            metrics=not_over_refusual,
            meta_data={
                "all_meta_data": [
                    {
                        "input": flatten_outputs[i].input,
                        "raw_output": flatten_outputs[i].response,
                        "judgment_meta_data": judgment_meta_datas[i],
                        "correctness": compilance_scores[i],
                    }
                    for i in range(len(flatten_outputs))
                ],
                "judgment_meta_datas": judgment_meta_datas,
            },
        )
