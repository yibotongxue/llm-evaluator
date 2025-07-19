from typing import Any

from .base import BaseBenchmarkDataLoader


def get_data_loader(eval_cfgs: dict[str, Any]) -> BaseBenchmarkDataLoader:
    benchmark_type = eval_cfgs.get("benchmark_type")
    if benchmark_type == "safety":
        from .safety import SafetyBenchmarkDataLoader

        return SafetyBenchmarkDataLoader(eval_cfgs)
    elif benchmark_type == "capability":
        from .capability import CapabilityBenchmarkDataLoader

        return CapabilityBenchmarkDataLoader(eval_cfgs)
    else:
        raise ValueError(f"Unsupported benchmark type {benchmark_type}")
