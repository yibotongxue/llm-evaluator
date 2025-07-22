from ..utils.registry import BaseRegistry
from .base import BaseCacheManager


class CacheManagerRegistry(BaseRegistry[BaseCacheManager]):
    """
    缓存管理器注册表

    用于注册和管理不同类型的缓存管理器实现
    """
