from abc import ABC, abstractmethod
from typing import Any


class BaseCacheManager(ABC):
    """
    缓存管理器的基类

    定义了缓存加载和保存的基本接口，所有具体的缓存实现都需继承此类

    参数
    ----
    cache_cfgs : dict[str, Any]
        缓存配置字典
    """

    def __init__(self, cache_cfgs: dict[str, Any]) -> None:
        self.cache_cfgs = cache_cfgs
        self.force_update = cache_cfgs.get("force_update", False)

    def load_cache(self, key: str) -> dict[str, Any] | None:
        """
        加载缓存内容

        如果强制更新设置为True，则始终返回None

        参数
        ----
        key : str
            缓存键名

        返回
        ----
        dict[str, Any] | None
            缓存的内容，如果不存在则返回None
        """
        if self.force_update:
            return None
        return self._load_cache(key)

    @abstractmethod
    def _load_cache(self, key: str) -> dict[str, Any] | None:
        """
        从缓存源加载数据的具体实现

        参数
        ----
        key : str
            缓存键名

        返回
        ----
        dict[str, Any] | None
            缓存的内容，如果不存在则返回None
        """

    @abstractmethod
    def save_cache(self, key: str, value: dict[str, Any]) -> None:
        """
        将数据保存到缓存

        参数
        ----
        key : str
            缓存键名
        value : dict[str, Any]
            要缓存的数据内容
        """
