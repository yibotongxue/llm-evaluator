from __future__ import annotations

from typing import Any, ClassVar

from ..utils.tools import dict_to_hash
from .base import BaseInference


class InferenceFactory:
    _instance: ClassVar[InferenceFactory | None] = None  # 单例实例引用
    _inference_pool: dict[str, BaseInference]  # 改为实例属性

    def __new__(cls) -> InferenceFactory:
        """单例模式实例控制"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化实例属性
            cls._instance._inference_pool = {}
        return cls._instance

    def get_inference_instance(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> BaseInference:
        """实例方法版本 (原类方法功能)"""
        cfgs_dict = {
            "model_cfgs": model_cfgs.copy(),
            "inference_cfgs": inference_cfgs.copy(),
        }
        cfgs_hash = dict_to_hash(cfgs_dict)

        for k, v in self._inference_pool.items():
            if not k == cfgs_hash:
                v.shutdown()

        if cfgs_hash in self._inference_pool:
            return self._inference_pool[cfgs_hash]

        backend = model_cfgs.get("inference_backend")
        if backend == "api":
            from .api_llm import get_api_llm_inference

            return get_api_llm_inference(
                cfgs_hash=cfgs_hash,
                model_cfgs=model_cfgs,
                inference_cfgs=inference_cfgs,
            )
        elif backend == "hf":
            from .hf import HuggingFaceInference

            return HuggingFaceInference(
                cfgs_hash=cfgs_hash,
                model_cfgs=model_cfgs,
                inference_cfgs=inference_cfgs,
            )
        elif backend == "vllm":
            from .vllm import VllmInference

            return VllmInference(
                cfgs_hash=cfgs_hash,
                model_cfgs=model_cfgs,
                inference_cfgs=inference_cfgs,
            )
        else:
            raise ValueError(f"Not supported inference backend: {backend}")
