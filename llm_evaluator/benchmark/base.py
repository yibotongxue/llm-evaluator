from abc import ABC, abstractmethod
from typing import Any

from ..data_loader import get_data_loader
from ..inference import InferenceFactory
from ..metrics import BaseMetricsComputer
from ..utils.json_utils import save_json
from ..utils.type_utils import (
    EvalConfigs,
    EvaluateResult,
    InferenceOutput,
    MetricsOutput,
    to_dict,
)


class BaseBenchmark(ABC):
    def __init__(
        self,
        eval_cfgs: dict[str, Any],
        model_cfgs: dict[str, Any],
        inference_cfgs: dict[str, Any],
        cache_cfgs: dict[str, Any] | None,
    ):
        self.eval_cfgs = EvalConfigs(**eval_cfgs)
        self.attack_cfgs = self.eval_cfgs.attack_cfgs
        self.model_cfgs = model_cfgs
        self.inference_cfgs = inference_cfgs
        self.inference_batch_size = inference_cfgs.pop("inference_batch_size", 32)
        self.model = InferenceFactory.get_inference_instance(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs, cache_cfgs=cache_cfgs
        )
        data_loader = get_data_loader(eval_cfgs=eval_cfgs)
        self.dataset = data_loader.load_dataset()
        self.metrics = self.init_metrics()

    @abstractmethod
    def init_metrics(self) -> dict[str, list[BaseMetricsComputer]]:
        pass

    def inference(self) -> dict[str, list[InferenceOutput]]:
        result: dict[str, list[InferenceOutput]] = {}
        for benchmark_name, inputs in self.dataset.items():
            result[benchmark_name] = self.model.generate(
                inputs,
                enable_tqdm=True,
                tqdm_args={"desc": f"Generating outputs for {benchmark_name}"},
            )
        return result

    def evaluate(self) -> dict[str, EvaluateResult]:
        output = self.inference()
        save_json(to_dict(output), "./output.json")
        result: dict[str, EvaluateResult] = {}
        for benchmark_name, outputs in output.items():
            metrics_result: list[MetricsOutput] = []
            for metrics_computer in self.metrics[benchmark_name]:
                metrics_output = metrics_computer.compute_metrics(outputs)
                metrics_result.append(metrics_output)
            result[benchmark_name] = EvaluateResult(
                metrics=metrics_result,
                benchmark_cfgs=self.eval_cfgs.benchmarks[benchmark_name],
                meta_data={"raw_output": outputs},
            )
        return result
