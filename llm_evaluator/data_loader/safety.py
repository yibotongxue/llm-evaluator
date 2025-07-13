from datasets import load_dataset

from ..data_formatter import SafetyDataFormatterRegistry
from ..utils.type_utils import BenchmarkConfigs, InferenceInput
from .base import BaseBenchmarkDataLoader


class SafetyBenchmarkDataLoader(BaseBenchmarkDataLoader):
    def load_benchmark_dataset(
        self, benchmark_cfgs: BenchmarkConfigs
    ) -> list[InferenceInput]:
        template = benchmark_cfgs.data_template
        data_formatter = SafetyDataFormatterRegistry.get_by_name(template)()
        split = benchmark_cfgs.model_dump().get("split", "train")
        dataset = load_dataset(benchmark_cfgs.data_name_or_path)[split]
        data_size = benchmark_cfgs.data_size
        raw_samples = [
            data_formatter.format_conversation(raw_sample) for raw_sample in dataset
        ]
        if data_size is not None:
            raw_samples = raw_samples[: min(data_size, len(dataset))]
        return raw_samples
