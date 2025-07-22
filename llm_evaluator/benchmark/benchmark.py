from typing import Any

from ..data.data_loader import BenchmarkDataLoader
from ..inference import InferenceFactory
from ..metrics import BaseMetricsComputer, get_metrics_computer
from ..utils.type_utils import (
    EvalConfigs,
    EvaluateResult,
    InferenceOutput,
    MetricsOutput,
)


class Benchmark:
    """
    基准测试类，用于评估LLM模型性能
    """

    def __init__(
        self,
        eval_cfgs: dict[str, Any],
        model_cfgs: dict[str, Any],
        inference_cfgs: dict[str, Any],
        cache_cfgs: dict[str, Any] | None = None,
    ):
        """
        初始化基准测试类

        参数
        ----
        eval_cfgs : dict[str, Any]
            评估配置信息
        model_cfgs : dict[str, Any]
            模型配置信息
        inference_cfgs : dict[str, Any]
            推理配置信息
        cache_cfgs : dict[str, Any] | None
            缓存配置信息，可选
        """
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
        """
        初始化评估指标

        根据评估配置创建相应的评估指标计算器
        """
        self.metrics: dict[str, list[BaseMetricsComputer]] = {}
        for benchmark_name, benchmark_cfgs in self.eval_cfgs.benchmarks.items():
            self.metrics[benchmark_name] = [
                get_metrics_computer(metrics_cfg)
                for metrics_cfg in benchmark_cfgs.metrics_cfgs
            ]

    def inference(self) -> dict[tuple[str, str], list[list[InferenceOutput]]]:
        """
        执行模型推理

        对每个基准测试的数据集进行模型推理，并返回推理结果

        返回
        ----
        dict[tuple[str, str], list[InferenceOutput]]
            包含每个基准测试和指标名称对应的推理结果列表的字典
        """
        inference_results: dict[tuple[str, str], list[list[InferenceOutput]]] = {}
        for benchmark_name, inputs in self.dataset.items():
            for metrics_computer in self.metrics[benchmark_name]:
                outputs = self.model.generate(
                    inputs,
                    **metrics_computer.infer_settings(),
                    enable_tqdm=True,
                    tqdm_args={"desc": f"Generating outputs for {benchmark_name}"},
                )
                inference_results[(benchmark_name, metrics_computer.metrics_name)] = (
                    outputs
                )
        return inference_results

    def evaluate(self) -> dict[str, EvaluateResult]:
        """
        执行评估过程

        对每个基准测试的数据集进行模型推理并计算评估指标

        返回
        ----
        dict[str, EvaluateResult]
            包含各基准测试评估结果的字典
        """
        outputs = self.inference()
        result: dict[str, EvaluateResult] = {}
        for benchmark_name in self.dataset.keys():
            metrics_result: list[MetricsOutput] = []
            for metrics_computer in self.metrics[benchmark_name]:
                output = outputs[(benchmark_name, metrics_computer.metrics_name)]
                metrics_output = metrics_computer.compute_metrics(output)
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
