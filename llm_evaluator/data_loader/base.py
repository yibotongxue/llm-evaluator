from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from datasets import load_dataset

from ..data_formatter import BaseDataFormatter
from ..utils.type_utils import BenchmarkConfigs, EvalConfigs, InferenceInput


class BaseBenchmarkDataLoader(ABC):
    def __init__(self, eval_cfgs: dict[str, Any]):
        self.eval_cfgs = EvalConfigs(**eval_cfgs)
        self.benchmarks = self.eval_cfgs.benchmarks

    def load_dataset(self) -> dict[str, list[InferenceInput]]:
        return {
            benchmark_name: self.load_benchmark_dataset(benchmark_name, benchmark_cfgs)
            for benchmark_name, benchmark_cfgs in self.benchmarks.items()
        }

    @cached_property
    def data_formatter_dict(self) -> dict[str, BaseDataFormatter]:
        return {
            benchmark_name: self.get_data_formatter(benchmark_cfgs)
            for benchmark_name, benchmark_cfgs in self.benchmarks.items()
        }

    def load_benchmark_dataset(
        self, benchmark_name: str, benchmark_cfgs: BenchmarkConfigs
    ) -> list[InferenceInput]:
        data_cfgs = benchmark_cfgs.data_cfgs

        data_formatter = self.data_formatter_dict[benchmark_name]

        # 数据大小
        data_size = data_cfgs.get("data_size")

        # 任务列表
        task_list = data_cfgs.get("task_list")

        # 加载数据
        data_path = data_cfgs["data_path"]
        load_cfgs = data_cfgs.get("load_cfgs", {})
        dataset = load_dataset(path=data_path, **load_cfgs)
        raw_samples = [
            data_formatter.format_conversation(raw_sample)
            for raw_sample in dataset
            if task_list is None
            or data_formatter.is_in_task_list(raw_sample, task_list)
        ]

        if data_size is not None:
            raw_samples = raw_samples[: min(data_size, len(raw_samples))]
        return raw_samples

    @abstractmethod
    def get_data_formatter(self, benchmark_cfgs: BenchmarkConfigs) -> BaseDataFormatter:
        pass
