from abc import ABC, abstractmethod
from typing import Any


class BaseCacheManager(ABC):
    def __init__(self, cache_cfgs: dict[str, Any]) -> None:
        self.cache_cfgs = cache_cfgs

    @abstractmethod
    def load_cache(self, key: str) -> dict[str, Any] | None:
        pass

    @abstractmethod
    def save_cache(self, key: str, value: dict[str, Any]) -> None:
        pass
