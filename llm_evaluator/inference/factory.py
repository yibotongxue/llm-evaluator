from typing import Any

from .api_llm import get_api_llm_inference
from .base import BaseInference
from .hf import HuggingFaceInference


def get_inference(
    model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
) -> BaseInference:
    backend = model_cfgs.get("inference_backend")
    if backend == "api":
        return get_api_llm_inference(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs
        )
    elif backend == "hf":
        return HuggingFaceInference(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs
        )
    else:
        raise ValueError(f"Not supported inference backend: {backend}")
