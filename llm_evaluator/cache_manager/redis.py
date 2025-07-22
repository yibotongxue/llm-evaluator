import json
from typing import Any

import redis  # type: ignore [import-untyped]

from .base import BaseCacheManager
from .registry import CacheManagerRegistry


@CacheManagerRegistry.register("redis")
class RedisCacheManager(BaseCacheManager):
    """
    基于Redis的缓存管理器

    将缓存内容存储在Redis数据库中

    参数
    ----
    cache_cfgs : dict[str, Any]
        Redis连接配置，直接传递给redis.Redis构造函数
    """

    def __init__(self, cache_cfgs: dict[str, Any]) -> None:
        super().__init__(cache_cfgs=cache_cfgs)
        self.redis_client = redis.Redis(**cache_cfgs)

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        """
        从Redis加载缓存数据

        参数
        ----
        key : str
            缓存键名

        返回
        ----
        dict[str, Any] | None
            缓存的内容，如果不存在则返回None
        """
        value = self.redis_client.get(key)
        if value is not None:
            # TODO 需要更仔细地检查加载的内容
            return json.loads(value)  # type: ignore [no-any-return]
        return None

    def save_cache(self, key: str, value: dict[str, Any]) -> None:
        """
        保存数据到Redis缓存

        参数
        ----
        key : str
            缓存键名
        value : dict[str, Any]
            要缓存的数据内容
        """
        self.redis_client.set(key, json.dumps(value))
