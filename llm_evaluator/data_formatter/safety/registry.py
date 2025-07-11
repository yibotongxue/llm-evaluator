from collections.abc import Callable

from .base import BaseSafetyDataFormatter


class SafetyDataFormatterRegistry:
    """
    安全数据处理注册类
    """

    _registry: dict[str, type[BaseSafetyDataFormatter]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[BaseSafetyDataFormatter]], type[BaseSafetyDataFormatter]]:
        def decorator(
            processor_cls: type[BaseSafetyDataFormatter],
        ) -> type[BaseSafetyDataFormatter]:
            cls._registry[name] = processor_cls
            return processor_cls

        return decorator

    @classmethod
    def get_by_name(cls, name: str) -> type[BaseSafetyDataFormatter]:
        return cls._registry[name]
