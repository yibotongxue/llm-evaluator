from pathlib import Path
from typing import Any

from ..utils.json_utils import load_json, save_json
from ..utils.logger import Logger
from .base import BaseCacheManager
from .registry import CacheManagerRegistry


@CacheManagerRegistry.register("json_file")
class JSONFileCacheManager(BaseCacheManager):
    """
    基于JSON文件的缓存管理器

    将缓存内容以JSON文件形式存储在指定目录中，并保持内存缓存以提高读取速度

    参数
    ----
    cache_cfgs : dict[str, Any]
        缓存配置，包含以下可选项：
        - cache_dir: 缓存目录路径，默认为"./cache"
        - flush_threshold: 达到多少脏数据时写入磁盘，默认为10
    """

    def __init__(self, cache_cfgs: dict[str, Any]) -> None:
        super().__init__(cache_cfgs)
        self.cache_dir = Path(cache_cfgs.get("cache_dir", "./cache"))
        self.flush_threshold = cache_cfgs.get("flush_threshold", 10)
        self.logger = Logger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._dirty_keys: set[str] = set()

        self._load_all_from_disk()

    def _safe_key(self, key: str) -> str:
        """
        将键名转换为安全的文件名

        参数
        ----
        key : str
            原始键名

        返回
        ----
        str
            安全的文件名
        """
        return key.replace("/", "_")  # 简单处理非法文件名字符

    def _get_file_path(self, key: str) -> Path:
        """
        获取缓存键对应的文件路径

        参数
        ----
        key : str
            缓存键名

        返回
        ----
        Path
            对应的文件路径
        """
        return self.cache_dir / f"{self._safe_key(key)}.json"

    def _load_all_from_disk(self) -> None:
        """
        从磁盘加载所有缓存文件到内存

        遍历缓存目录中的所有JSON文件并将其加载到内存中
        """
        for file in self.cache_dir.glob("*.json"):
            key = file.stem
            try:
                self._memory_cache[key] = load_json(str(file))
            except Exception as err:
                self.logger.warning(
                    f"读取文件{str(file)}的时候出现异常，异常信息为{err}"
                )
                continue  # 忽略损坏的文件

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        """
        从内存缓存中加载数据

        参数
        ----
        key : str
            缓存键名

        返回
        ----
        dict[str, Any] | None
            缓存的内容，如果不存在则返回None
        """
        return self._memory_cache.get(key)

    def save_cache(self, key: str, value: dict[str, Any]) -> None:
        """
        保存数据到缓存

        将数据保存到内存缓存，并在达到阈值时刷新到磁盘

        参数
        ----
        key : str
            缓存键名
        value : dict[str, Any]
            要缓存的数据内容
        """
        self._memory_cache[key] = value
        self._dirty_keys.add(key)

        if len(self._dirty_keys) >= self.flush_threshold:
            self._flush_dirty_to_disk()

    def _flush_dirty_to_disk(self) -> None:
        """
        将修改过的缓存数据写入磁盘

        遍历所有标记为"脏"的键，将其对应的数据保存到磁盘文件中
        """
        for key in list(self._dirty_keys):
            path = self._get_file_path(key)
            save_json(self._memory_cache[key], str(path))
        self._dirty_keys.clear()

    def __del__(self) -> None:
        """
        对象销毁时的清理操作

        确保所有未保存的缓存数据都写入磁盘
        """
        self._flush_dirty_to_disk()
