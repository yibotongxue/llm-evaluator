from typing import Any

from .base import BaseInference


def get_inference(
    model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
) -> BaseInference:
    backend = model_cfgs.get("inference_backend")
    if backend == "api":
        from .api_llm import get_api_llm_inference

        return get_api_llm_inference(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs
        )
    elif backend == "hf":
        from .hf import HuggingFaceInference

        return HuggingFaceInference(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs
        )
    else:
        raise ValueError(f"Not supported inference backend: {backend}")
