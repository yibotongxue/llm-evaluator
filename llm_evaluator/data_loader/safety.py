from datasets import load_dataset

from ..data_formatter import SafetyDataFormatterRegistry
from ..utils.type_utils import BenchmarkConfigs, InferenceInput
from .base import BaseBenchmarkDataLoader


class SafetyBenchmarkDataLoader(BaseBenchmarkDataLoader):
    def load_benchmark_dataset(
        self, benchmark_cfgs: BenchmarkConfigs
    ) -> list[InferenceInput]:
        data_cfgs = benchmark_cfgs.data_cfgs

        # 格式化模板
        template = data_cfgs.pop("data_template")
        data_formatter = SafetyDataFormatterRegistry.get_by_name(template)()

        # 数据大小
        data_size = data_cfgs.pop("data_size")

        # 任务列表
        task_list = data_cfgs.pop("task_list")

        # 加载数据
        data_path = data_cfgs.pop("data_path")
        data_name = data_cfgs.pop("data_name", None)
        dataset = load_dataset(path=data_path, name=data_name, **data_cfgs)
        raw_samples = [
            data_formatter.format_conversation(raw_sample)
            for raw_sample in dataset
            if task_list is None
            or data_formatter.is_in_task_list(raw_sample, task_list)
        ]

        if data_size is not None:
            raw_samples = raw_samples[: min(data_size, len(raw_samples))]
        return raw_samples
