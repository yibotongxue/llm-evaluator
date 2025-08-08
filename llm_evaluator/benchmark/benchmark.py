from typing import Any

from ..data.data_loader import BenchmarkDataLoader
from ..inference import InferenceFactory
from ..metrics import BaseMetricsComputer, get_metrics_computer
from ..prompts import PromptBuilderRegistry
from ..prompts.attack import AttackPromptBuilder
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
        self.model_cfgs = model_cfgs
        self.inference_cfgs = inference_cfgs
        self.model = InferenceFactory.get_inference_instance(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs, cache_cfgs=cache_cfgs
        )
        data_loader = BenchmarkDataLoader(eval_cfgs=eval_cfgs)
        self.dataset = data_loader.load_dataset()
        self.data_formatter_dict = data_loader.data_formatter_dict
        self.init_safety()
        self.init_metrics()

    def init_safety(self) -> None:
        self.attack_cfgs = self.eval_cfgs.attack_cfgs
        self.benchmark_type = self.eval_cfgs.benchmark_type
        if self.benchmark_type != "safety" and self.attack_cfgs is not None:
            raise ValueError("不能在非安全评估中设置攻击配置")
        self.prompt_builder_types: list[tuple[str, str | dict[str, Any] | None]] = [
            ("None", None)
        ]
        if self.benchmark_type == "safety":
            if self.attack_cfgs is None:
                self.attack_cfgs = []
            for attack_cfg in self.attack_cfgs:
                if attack_cfg["attack_type"] == "prompt_builder":
                    prompt_builder_cfgs = attack_cfg["prompt_builder_cfgs"]
                    if isinstance(prompt_builder_cfgs, dict):
                        if not "name" in prompt_builder_cfgs.keys():
                            raise ValueError(
                                "攻击配置中的提示词构建器参数中必须包含name"
                            )
                        prompt_builder_name = prompt_builder_cfgs["name"]
                        if not isinstance(prompt_builder_name, str):
                            raise ValueError(
                                "攻击配置中的提示词构建器参数name必须为字符串"
                            )
                        for k in prompt_builder_cfgs.keys():
                            if not isinstance(k, str):
                                raise ValueError(
                                    f"攻击配置中的提示词构建器参数名必须为字符串，得到一个{k}"
                                )
                    elif isinstance(prompt_builder_cfgs, str):
                        prompt_builder_name = prompt_builder_cfgs
                    else:
                        raise TypeError(f"提示词配置必须为字符串或字典")
                    PromptBuilderRegistry.verify_type(
                        prompt_builder_name, AttackPromptBuilder  # type: ignore [type-abstract]
                    )
                    self.prompt_builder_types.append(
                        (attack_cfg["attack_name"], prompt_builder_cfgs)
                    )

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

    def inference(
        self,
    ) -> dict[str, dict[tuple[str, str], list[list[InferenceOutput]]]]:
        """
        执行模型推理

        对每个基准测试的数据集进行模型推理，并返回推理结果

        返回
        ----
        dict[tuple[str, str], list[InferenceOutput]]
            包含每个基准测试和指标名称对应的推理结果列表的字典
        """
        inference_results: dict[
            str, dict[tuple[str, str], list[list[InferenceOutput]]]
        ] = {}
        for attack_name, prompt_builder_cfgs in self.prompt_builder_types:
            inference_results[attack_name] = {}
            for benchmark_name, inputs in self.dataset.items():
                for metrics_computer in self.metrics[benchmark_name]:
                    infer_settings = metrics_computer.infer_settings()
                    if (
                        infer_settings["prompt_template"] is not None
                        and self.benchmark_type == "safety"
                    ):
                        raise ValueError(
                            "安全评估的推理不能在度量计算设置提示词模板，必须在攻击配置中设置提示词模板"
                        )
                    if prompt_builder_cfgs is not None:
                        infer_settings["prompt_template"] = prompt_builder_cfgs
                    outputs = self.model.generate(
                        inputs,
                        **infer_settings,
                        enable_tqdm=True,
                        tqdm_args={
                            "desc": f"Generating outputs for {benchmark_name} with {attack_name}"
                        },
                    )
                    inference_results[attack_name][
                        (benchmark_name, metrics_computer.metrics_name)
                    ] = outputs
        return inference_results

    def evaluate(self) -> dict[str, dict[str, EvaluateResult]]:
        """
        执行评估过程

        对每个基准测试的数据集进行模型推理并计算评估指标

        返回
        ----
        dict[str, EvaluateResult]
            包含各基准测试评估结果的字典
        """
        outputs = self.inference()
        result: dict[str, dict[str, EvaluateResult]] = {}
        for attack_name in outputs.keys():
            result[attack_name] = {}
            for benchmark_name in self.dataset.keys():
                metrics_result: list[MetricsOutput] = []
                for metrics_computer in self.metrics[benchmark_name]:
                    output = outputs[attack_name][
                        (benchmark_name, metrics_computer.metrics_name)
                    ]
                    metrics_output = metrics_computer.compute_metrics(output)
                    metrics_result.append(metrics_output)
                result[attack_name][benchmark_name] = EvaluateResult(
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
