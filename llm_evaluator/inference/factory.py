from typing import Any

from ..cache_manager import get_cache_manager
from ..utils.tools import dict_to_hash
from .base import BaseInference, InferenceInterface
from .cached import CachedInference


class InferenceFactory:
    _inference_pool: dict[str, BaseInference] = {}

    @classmethod
    def get_inference_instance(
        cls,
        model_cfgs: dict[str, Any],
        inference_cfgs: dict[str, Any],
        cache_cfgs: dict[str, Any] | None,
    ) -> InferenceInterface:
        instance = cls._get_inference_instance(
            model_cfgs=model_cfgs, inference_cfgs=inference_cfgs
        )
        if cache_cfgs is None:
            return instance
        cache_manager = get_cache_manager(cache_cfgs=cache_cfgs)
        return CachedInference(inference=instance, cache_manager=cache_manager)

    @classmethod
    def _get_inference_instance(
        cls,
        model_cfgs: dict[str, Any],
        inference_cfgs: dict[str, Any],
        cached_cfgs: dict[str, Any] | None = None,
    ) -> BaseInference:
        cfgs_dict = {
            "model_cfgs": model_cfgs.copy(),
            "inference_cfgs": inference_cfgs.copy(),
        }
        cfgs_hash = dict_to_hash(cfgs_dict)

        for k, v in cls._inference_pool.items():
            if not k == cfgs_hash:
                v.shutdown()

        if cfgs_hash in cls._inference_pool:
            return cls._inference_pool[cfgs_hash]

        backend = model_cfgs.get("inference_backend")

        instance: BaseInference | None = None

        if backend == "api":
            from .api_llm import get_api_llm_inference

            instance = get_api_llm_inference(
                cfgs_hash=cfgs_hash,
                model_cfgs=model_cfgs,
                inference_cfgs=inference_cfgs,
            )
        elif backend == "hf":
            from .hf import HuggingFaceInference

            instance = HuggingFaceInference(
                cfgs_hash=cfgs_hash,
                model_cfgs=model_cfgs,
                inference_cfgs=inference_cfgs,
            )
        elif backend == "vllm":
            from .vllm import VllmInference

            instance = VllmInference(
                cfgs_hash=cfgs_hash,
                model_cfgs=model_cfgs,
                inference_cfgs=inference_cfgs,
            )
        else:
            raise ValueError(f"Not supported inference backend: {backend}")
        cls._inference_pool[cfgs_hash] = instance

        return instance
