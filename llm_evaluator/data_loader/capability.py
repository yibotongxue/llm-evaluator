from typing import Any

from ..data_formatter import BaseDataFormatter
from ..data_formatter.capability import CapabilityDataFormatterRegistry
from ..prompts import PromptBuilderRegistry
from ..utils.type_utils import BenchmarkConfigs
from .base import BaseBenchmarkDataLoader


class CapabilityBenchmarkDataLoader(BaseBenchmarkDataLoader):
    def get_data_formatter(self, benchmark_cfgs: BenchmarkConfigs) -> BaseDataFormatter:
        data_cfgs = benchmark_cfgs.data_cfgs
        template_cfgs: dict[str, Any] = data_cfgs.get("template_cfgs", {})

        prompt_builder_type: str = template_cfgs.get("prompt_builder_name", "default")
        prompt_builder = PromptBuilderRegistry.get_by_name(prompt_builder_type)()

        template: str = data_cfgs.get("data_template", "default")
        data_formatter = CapabilityDataFormatterRegistry.get_by_name(template)(
            prompt_builder
        )
        return data_formatter
