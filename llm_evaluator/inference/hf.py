from typing import Any

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ..utils.logger import Logger
from ..utils.type_utils import InferenceInput, InferenctOutput
from .base import BaseInference

__all__ = [
    "HuggingFaceInference",
]

_logger = Logger(__name__)


class HuggingFaceInference(BaseInference):
    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        super().__init__(model_cfgs=model_cfgs, inference_cfgs=inference_cfgs)
        model_name = self.model_cfgs["model_name_or_path"]
        _logger.info(f"预备加载模型{model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        _logger.info(f"模型{model_name}已加载")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name
        )
        _logger.info(f"分词器{model_name}已加载")
        self.accelerator = Accelerator()
        self.model: PreTrainedModel = self.accelerator.prepare(model)
        _logger.info(f"使用加速设备{self.accelerator.device}")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, inputs: list[InferenceInput]) -> list[InferenctOutput]:
        tokenize_cfgs = self.inference_cfgs.get("tokenize_cfgs", {})
        generate_cfgs = self.inference_cfgs.get("generate_cfgs", {})
        messages = [input.conversation for input in inputs]
        encoded_inputs = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            **tokenize_cfgs,
        )
        outputs = self.model.generate(
            input_ids=encoded_inputs["input_ids"].to(self.accelerator.device),
            attention_mask=encoded_inputs["attention_mask"].to(self.accelerator.device),
            num_return_sequences=1,
            **generate_cfgs,
        )
        output_ids = [
            output[encoded_inputs["input_ids"].shape[-1] :] for output in outputs
        ]
        responses = [
            self.tokenizer.decode(output_id, skip_special_tokens=True)
            for output_id in output_ids
        ]
        inference_outptus = [
            InferenctOutput(
                response=responses[idx],
                input=inputs[idx].model_dump(),
                engine="hf",
                meta_data={
                    "output_id": output_ids[idx].tolist(),
                    "tokenize_cfgs": tokenize_cfgs,
                    "generate_cfgs": generate_cfgs,
                },
            )
            for idx in range(len(inputs))
        ]
        return inference_outptus


def main() -> None:
    import argparse

    from ..utils.config import load_config, update_config_with_unparsed_args

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

    inference = HuggingFaceInference(
        model_cfgs=cfgs["model_cfgs"], inference_cfgs=cfgs["inference_cfgs"]
    )

    inference_input = [
        InferenceInput.from_prompts(
            prompt="中国的首都是哪里？",
            system_prompt="你是一个人工智能助手",
        ),
        InferenceInput.from_prompts(
            prompt="Where is the capital of China?",
            system_prompt="You are an AI assistant",
        ),
        InferenceInput.from_prompts(
            prompt="中国有多少个省份？分别是哪些？",
            system_prompt="你是一个人工智能助手",
        ),
        InferenceInput.from_prompts(
            prompt="How many provinces are there in China? What are they?",
            system_prompt="You are an AI assistant",
        ),
    ]

    outputs = inference.generate(inference_input)

    for inference_output in outputs:
        print(inference_output.response)


if __name__ == "__main__":
    main()
