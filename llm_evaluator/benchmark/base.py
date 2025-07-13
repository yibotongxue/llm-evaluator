from abc import ABC, abstractmethod
from typing import Any

from ..data_loader import get_data_loader
from ..inference import get_inference
from ..metrics import BaseMetricsComputer
from ..utils.type_utils import EvalConfigs, EvaluateResult, InferenceOutput


class BaseBenchmark(ABC):
    def __init__(
        self,
        eval_cfgs: dict[str, Any],
        model_cfgs: dict[str, Any],
        inference_cfgs: dict[str, Any],
    ):
        self.eval_cfgs = EvalConfigs(**eval_cfgs)
        self.attack_cfgs = self.eval_cfgs.attack_cfgs
        self.model_cfgs = model_cfgs
        self.inference_cfgs = inference_cfgs
        self.inference_batch_size = inference_cfgs.pop("inference_batch_size", 32)
        self.model = get_inference(model_cfgs=model_cfgs, inference_cfgs=inference_cfgs)
        data_loader = get_data_loader(eval_cfgs=eval_cfgs)
        self.dataset = data_loader.load_dataset()
        self.metrics = self.init_metrics()

    @abstractmethod
    def init_metrics(self) -> list[BaseMetricsComputer]:
        pass

    def inference(self) -> dict[str, list[InferenceOutput]]:
        result: dict[str, list[InferenceOutput]] = {}
        for benchmark_name, inputs in self.dataset.items():
            result[benchmark_name] = self.model.generate(inputs)
        return result

    def evaluate(self) -> dict[str, EvaluateResult]:
        output = self.inference()
        result: dict[str, EvaluateResult] = {}
        for benchmark_name, outputs in output.items():
            metrics = self.compute_metrics(outputs)
            result[benchmark_name] = metrics
        return result

    def compute_metrics(self, outputs: list[InferenceOutput]) -> EvaluateResult:
        metrics_result: dict[str, float] = {}
        for metrics_computer in self.metrics:
            metrics_name, metrics_value = metrics_computer.compute_metrics(outputs)
            metrics_result[metrics_name] = metrics_value
        return EvaluateResult(
            metrics=metrics_result,
            benchmark_cfgs=self.eval_cfgs.benchmarks,
            raw_output=outputs,
        )
