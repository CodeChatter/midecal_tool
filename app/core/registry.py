"""通用注册表 — 插件化架构基础设施"""

from typing import Any, Callable, Dict, List


class Registry:
    """通用工厂注册表，支持装饰器注册和按 key 创建实例。"""

    def __init__(self, name: str):
        self.name = name
        self._factories: Dict[str, Callable[..., Any]] = {}

    def register(self, key: str):
        """装饰器：注册一个类（类本身作为工厂）。"""
        def decorator(cls):
            self._factories[key] = cls
            return cls
        return decorator

    def register_factory(self, key: str, fn: Callable[..., Any]):
        """手动注册一个工厂函数。"""
        self._factories[key] = fn

    def create(self, key: str, **kwargs) -> Any:
        """按 key 创建实例。"""
        if key not in self._factories:
            raise KeyError(
                f"[{self.name}] 未注册的 key: '{key}'，"
                f"可选: {self.available()}"
            )
        return self._factories[key](**kwargs)

    def available(self) -> List[str]:
        """返回所有已注册的 key。"""
        return list(self._factories.keys())


ocr_registry = Registry("ocr")
detector_registry = Registry("detector")
masking_registry = Registry("masking")
