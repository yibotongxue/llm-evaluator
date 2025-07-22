from ...utils.registry import BaseRegistry
from .base import BaseDataFormatter


class DataFormatterRegistry(BaseRegistry[BaseDataFormatter]):
    """
    数据格式化器注册表。

    用于管理和注册各种数据格式化器，以便根据名称动态获取相应的格式化器实例。
    """
