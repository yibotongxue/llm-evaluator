from functools import cached_property
from typing import Any

from datasets import load_dataset

from ..utils.type_utils import BenchmarkConfigs, EvalConfigs, InferenceInput
from .data_formatter import BaseDataFormatter, DataFormatterRegistry


class BenchmarkDataLoader:
    """
    基准测试数据加载器。

    用于加载和格式化基准测试数据集，以便进行模型评估。
    """

    def __init__(self, eval_cfgs: dict[str, Any]):
        """
        初始化基准测试数据加载器。

        参数
        ----------
        eval_cfgs : dict[str, Any]
            评估配置字典
        """
        self.eval_cfgs = EvalConfigs(**eval_cfgs)
        self.benchmarks = self.eval_cfgs.benchmarks

    def load_dataset(self) -> dict[str, list[InferenceInput]]:
        """
        加载所有基准测试数据集。

        返回
        -------
        dict[str, list[InferenceInput]]
            包含所有基准测试数据的字典，键为基准测试名称，值为推理输入列表
        """
        return {
            benchmark_name: self.load_benchmark_dataset(benchmark_name, benchmark_cfgs)
            for benchmark_name, benchmark_cfgs in self.benchmarks.items()
        }

    @cached_property
    def data_formatter_dict(self) -> dict[str, BaseDataFormatter]:
        """
        获取数据格式化器字典。

        返回
        -------
        dict[str, BaseDataFormatter]
            包含所有数据格式化器的字典，键为基准测试名称，值为对应的数据格式化器
        """
        return {
            benchmark_name: self.get_data_formatter(benchmark_cfgs)
            for benchmark_name, benchmark_cfgs in self.benchmarks.items()
        }

    def load_benchmark_dataset(
        self, benchmark_name: str, benchmark_cfgs: BenchmarkConfigs
    ) -> list[InferenceInput]:
        """
        加载特定基准测试数据集。

        参数
        ----------
        benchmark_name : str
            基准测试名称
        benchmark_cfgs : BenchmarkConfigs
            基准测试配置

        返回
        -------
        list[InferenceInput]
            推理输入列表
        """
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
            if (
                task_list is None
                or data_formatter.is_in_task_list(raw_sample, task_list)
            )
            and data_formatter.is_valid_sample(raw_sample)
        ]

        if data_size is not None:
            raw_samples = raw_samples[: min(data_size, len(raw_samples))]
        return raw_samples

    def get_data_formatter(self, benchmark_cfgs: BenchmarkConfigs) -> BaseDataFormatter:
        """
        获取数据格式化器。

        参数
        ----------
        benchmark_cfgs : BenchmarkConfigs
            基准测试配置

        返回
        -------
        BaseDataFormatter
            数据格式化器实例
        """
        data_cfgs = benchmark_cfgs.data_cfgs

        template: str = data_cfgs.get("data_template", "default")
        data_formatter = DataFormatterRegistry.get_by_name(template)()
        return data_formatter
