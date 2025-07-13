from abc import ABC, abstractmethod
from typing import Any

from ..utils.type_utils import BenchmarkConfigs, EvalConfigs, InferenceInput


class BaseBenchmarkDataLoader(ABC):
    def __init__(self, eval_cfgs: dict[str, Any]):
        self.eval_cfgs = EvalConfigs(**eval_cfgs)
        self.benchmarks = self.eval_cfgs.benchmarks

    def load_dataset(self) -> dict[str, list[InferenceInput]]:
        return {
            benchmark_name: self.load_benchmark_dataset(benchmark_cfgs)
            for benchmark_name, benchmark_cfgs in self.benchmarks.items()
        }

    @abstractmethod
    def load_benchmark_dataset(
        self, benchmark_cfgs: BenchmarkConfigs
    ) -> list[InferenceInput]:
        pass
