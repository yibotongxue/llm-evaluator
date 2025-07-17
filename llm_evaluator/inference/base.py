from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, ContextManager

from ..utils.shutdownable import Shutdownable
from ..utils.tools import dict_to_hash
from ..utils.type_utils import InferenceInput, InferenceOutput


class InferenceInterface(ABC):
    @abstractmethod
    def generate(
        self,
        inputs: list[InferenceInput],
        enable_tqdm: bool = False,
        tqdm_args: dict[str, Any] | None = None,
    ) -> list[InferenceOutput]:
        pass

    def update_inference_cfgs(
        self, new_inference_cfgs: dict[str, Any]
    ) -> ContextManager[InferenceInterface]:
        return self._TempConfigUpdater(self, new_inference_cfgs)

    @abstractmethod
    def _update_inference_cfgs(
        self, new_inference_cfgs: dict[str, Any]
    ) -> Callable[[], None]:
        pass

    class _TempConfigUpdater(AbstractContextManager["InferenceInterface"]):
        """内部上下文管理器类，处理配置的临时更新和恢复"""

        def __init__(
            self, owner: InferenceInterface, new_inference_cfgs: dict[str, Any]
        ):
            self.owner = owner  # 主类实例
            self.new_inference_cfgs = new_inference_cfgs  # 要更新的配置
            self.restore_func: Callable[[], None] | None = None

        def __enter__(self) -> InferenceInterface:
            """进入上下文时：备份原始配置并应用新配置"""
            self.restore_func = self.owner._update_inference_cfgs(
                self.new_inference_cfgs
            )

            return self.owner  # 返回主类实例以便链式调用

        def __exit__(self, exc_type, exc_value, traceback):  # type: ignore [no-untyped-def]
            """退出上下文时：恢复原始配置"""
            # 无论是否发生异常都恢复配置
            if self.restore_func is not None:
                self.restore_func()
            # 不处理异常，返回 None 让异常正常传播


class BaseInference(InferenceInterface, Shutdownable):
    def __init__(
        self, model_cfgs: dict[str, Any], inference_cfgs: dict[str, Any]
    ) -> None:
        cfgs_dict = {
            "model_cfgs": model_cfgs,
            "inference_cfgs": inference_cfgs,
        }
        self._cfgs_hash = dict_to_hash(cfgs_dict)
        self.model_cfgs = model_cfgs
        self.inference_cfgs = inference_cfgs

    def _update_inference_cfgs(
        self, new_inference_cfgs: dict[str, Any]
    ) -> Callable[[], None]:
        self.original_inference_cfgs = self.inference_cfgs.copy()
        self.original_cfgs_hash = self._cfgs_hash
        self.inference_cfgs.update(new_inference_cfgs)
        self._cfgs_hash = dict_to_hash(
            {
                "model_cfgs": self.model_cfgs,
                "inference_cfgs": self.inference_cfgs,
            }
        )

        def _restore_inference_cfgs() -> None:
            """Restore the original inference configurations."""
            self.inference_cfgs = self.original_inference_cfgs
            self._cfgs_hash = self.original_cfgs_hash

        return _restore_inference_cfgs

    @property
    def cfgs_hash(self) -> str:
        return self._cfgs_hash
