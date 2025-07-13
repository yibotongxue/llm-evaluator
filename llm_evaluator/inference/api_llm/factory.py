from typing import Any

from ...utils.tools import load_api_key
from .base import BaseApiLLMInference


def get_api_llm_inference(
    model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
) -> BaseApiLLMInference:
    model_sdk_type = model_cfgs.pop("model_sdk_type")
    model_cfgs = load_api_key(model_cfgs)
    if model_sdk_type == "openai":
        from .openai_api import OpenAIApiLLMInference

        return OpenAIApiLLMInference(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs
        )
    elif model_sdk_type == "gemini":
        from .gemini import GeminiApiLLMInference

        return GeminiApiLLMInference(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs
        )
    elif model_sdk_type == "anthropic":
        from .anthropic import AnthropicApiLLMInference

        return AnthropicApiLLMInference(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs
        )
    else:
        raise ValueError(f"模型类型{model_sdk_type}不被支持")
