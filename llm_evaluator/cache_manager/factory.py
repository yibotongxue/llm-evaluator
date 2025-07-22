from typing import Any

from .base import BaseCacheManager
from .registry import CacheManagerRegistry


def get_cache_manager(cache_cfgs: dict[str, Any]) -> BaseCacheManager:
    """
    获取缓存管理器实例的工厂函数

    根据配置中指定的缓存类型创建相应的缓存管理器实例

    参数
    ----
    cache_cfgs : dict[str, Any]
        缓存配置字典，必须包含'cache_type'键以指定缓存类型

    返回
    ----
    BaseCacheManager
        创建的缓存管理器实例

    异常
    ----
    ValueError
        当未指定cache_type时抛出
    """
    cache_type = cache_cfgs.pop("cache_type", None)
    if cache_type is None:
        raise ValueError("The cache type should be set")
    return CacheManagerRegistry.get_by_name(cache_type)(cache_cfgs=cache_cfgs)
