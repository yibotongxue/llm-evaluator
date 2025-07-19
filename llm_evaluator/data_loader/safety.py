from ..data_formatter import BaseDataFormatter
from ..data_formatter.safety import SafetyDataFormatterRegistry
from ..utils.type_utils import BenchmarkConfigs
from .base import BaseBenchmarkDataLoader


class SafetyBenchmarkDataLoader(BaseBenchmarkDataLoader):
    def get_data_formatter(self, benchmark_cfgs: BenchmarkConfigs) -> BaseDataFormatter:
        data_cfgs = benchmark_cfgs.data_cfgs

        template: str = data_cfgs.get("data_template", "default")
        data_formatter = SafetyDataFormatterRegistry.get_by_name(template)()
        return data_formatter
