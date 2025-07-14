from ..metrics import BaseMetricsComputer
from ..metrics.safety import get_safety_metrics
from .base import BaseBenchmark


class SafetyBenchmark(BaseBenchmark):
    def init_metrics(self) -> dict[str, list[BaseMetricsComputer]]:
        result: dict[str, list[BaseMetricsComputer]] = {}
        for benchmark_name, benchmark_cfgs in self.eval_cfgs.benchmarks.items():
            result[benchmark_name] = [
                get_safety_metrics(metrics_cfg)
                for metrics_cfg in benchmark_cfgs.metrics_cfgs
            ]
        return result


def main() -> None:
    import argparse

    from ..utils.config import load_config, update_config_with_unparsed_args
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

    print(cfgs)

    benchmark = SafetyBenchmark(**cfgs)

    result = benchmark.evaluate()
    save_json(to_dict(result), "./full_result.json")
    save_json(to_breif_dict(result), "./brief_result.json")


if __name__ == "__main__":
    main()
