from ..data_formatter.capability import BaseCapabilityDataFormatter
from ..metrics import BaseMetricsComputer
from ..metrics.capability import get_capability_metrics
from ..utils.type_utils import InferenceOutput
from .base import BaseBenchmark


class CapabilityBenchmark(BaseBenchmark):
    def inference(self) -> dict[str, list[InferenceOutput]]:
        result: dict[str, list[InferenceOutput]] = {}
        for benchmark_name, inputs in self.dataset.items():
            result[benchmark_name] = self.model.generate(
                inputs,
                enable_tqdm=True,
                tqdm_args={"desc": f"Generating outputs for {benchmark_name}"},
            )
            for output in result[benchmark_name]:
                data_formatter = self.data_formatter_dict[benchmark_name]
                if not isinstance(data_formatter, BaseCapabilityDataFormatter):
                    raise TypeError(
                        f"Data formatter for {benchmark_name} must be a "
                        f"BaseCapabilityDataFormatter, got {type(data_formatter).__name__}"
                    )
                output.extracted_answer = data_formatter.prompt_builder.extract_answer(
                    output.response
                )
        return result

    def init_metrics(self) -> dict[str, list[BaseMetricsComputer]]:
        result: dict[str, list[BaseMetricsComputer]] = {}
        for benchmark_name, benchmark_cfgs in self.eval_cfgs.benchmarks.items():
            result[benchmark_name] = [
                get_capability_metrics(metrics_cfg)
                for metrics_cfg in benchmark_cfgs.metrics_cfgs
            ]
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

    output_dir = cfgs["eval_cfgs"].pop("output_dir", "./output")

    save_json(cfgs, f"{output_dir}/cfgs.json")

    print(cfgs)

    benchmark = CapabilityBenchmark(**cfgs)

    result = benchmark.evaluate()
    save_json(to_dict(result), f"{output_dir}/full_result.json")
    save_json(to_breif_dict(result), f"{output_dir}/brief_result.json")


if __name__ == "__main__":
    main()
