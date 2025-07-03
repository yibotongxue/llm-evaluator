from llm_evaluator.utils.config import *
from llm_evaluator.utils.type_utils import InferenceInput

from .factory import get_api_llm_inference


def main() -> None:
    import argparse

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

    inference = get_api_llm_inference(cfgs["model_cfgs"], cfgs["inference_cfgs"])

    inference_input = [
        InferenceInput(
            prompt="中国的首都是哪里？",
            system_prompt="你是一个人工智能助手",
            meta_data={},
        ),
        InferenceInput(
            prompt="Where is the capital of China?",
            system_prompt="You are an AI assistant",
            meta_data={},
        ),
    ]

    inference_output = inference.generate(inference_input)

    print(inference_output[0].response)
    print(inference_output[1].response)


main()
