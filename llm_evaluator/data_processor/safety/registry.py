from collections.abc import Callable

from .base import BaseSafetyDataProcessor


class SafetyDataProcessorRegistry:
    """
    安全数据处理注册类
    """

    _registry: dict[str, type[BaseSafetyDataProcessor]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[BaseSafetyDataProcessor]], type[BaseSafetyDataProcessor]]:
        def decorator(
            processor_cls: type[BaseSafetyDataProcessor],
        ) -> type[BaseSafetyDataProcessor]:
            cls._registry[name] = processor_cls
            return processor_cls

        return decorator

    @classmethod
    def get_by_name(cls, name: str) -> type[BaseSafetyDataProcessor]:
        return cls._registry[name]
