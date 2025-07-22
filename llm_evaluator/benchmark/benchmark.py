from typing import Any

from ..data.data_loader import BenchmarkDataLoader
from ..inference import InferenceFactory
from ..metrics import BaseMetricsComputer, MetricsRegistry
from ..utils.type_utils import EvalConfigs, EvaluateResult, MetricsOutput


class Benchmark:
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
        self.model = InferenceFactory.get_inference_instance(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs, cache_cfgs=cache_cfgs
        )
        data_loader = BenchmarkDataLoader(eval_cfgs=eval_cfgs)
        self.dataset = data_loader.load_dataset()
        self.data_formatter_dict = data_loader.data_formatter_dict
        self.init_metrics()

    def init_metrics(self) -> None:
        self.metrics: dict[str, list[BaseMetricsComputer]] = {}
        for benchmark_name, benchmark_cfgs in self.eval_cfgs.benchmarks.items():
            self.metrics[benchmark_name] = [
                MetricsRegistry.get_by_name(metrics_cfg["metrics_type"])(metrics_cfg)
                for metrics_cfg in benchmark_cfgs.metrics_cfgs
            ]

    def evaluate(self) -> dict[str, EvaluateResult]:
        result: dict[str, EvaluateResult] = {}
        for benchmark_name, inputs in self.dataset.items():
            metrics_result: list[MetricsOutput] = []
            for metrics_computer in self.metrics[benchmark_name]:
                outputs = self.model.generate(
                    inputs,
                    **metrics_computer.infer_settings(),
                    enable_tqdm=True,
                    tqdm_args={"desc": f"Computing metrics for {benchmark_name}"},
                )
                metrics_output = metrics_computer.compute_metrics(outputs)
                metrics_result.append(metrics_output)
            result[benchmark_name] = EvaluateResult(
                metrics=metrics_result,
                benchmark_cfgs=self.eval_cfgs.benchmarks[benchmark_name],
                meta_data={},
            )
        return result


def main() -> None:
    import argparse

    from ..utils.config import (
        deepcopy_config,
        load_config,
        update_config_with_unparsed_args,
    )
    from ..utils.json_utils import save_json
    from ..utils.type_utils import to_breif_dict, to_dict

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file-path",
        type=str,
        required=True,
        help="The path to the config file",
    )
    args, unparsed_args = parser.parse_known_args()

    cfgs = load_config(args.config_file_path)
    update_config_with_unparsed_args(unparsed_args=unparsed_args, cfgs=cfgs)

    cfgs = deepcopy_config(cfgs)

    cfgs.pop("_common")

    output_dir = cfgs["eval_cfgs"].pop("output_dir", "./output")

    save_json(cfgs, f"{output_dir}/cfgs.json")

    print(cfgs)

    benchmark = Benchmark(**cfgs)

    result = benchmark.evaluate()
    save_json(to_dict(result), f"{output_dir}/full_result.json")
    save_json(to_breif_dict(result), f"{output_dir}/brief_result.json")


if __name__ == "__main__":
    main()
