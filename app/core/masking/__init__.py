from .solid import SolidColorStrategy
from ..registry import masking_registry

# 注册纯色工厂
masking_registry.register_factory("black", lambda: SolidColorStrategy((0, 0, 0)))
masking_registry.register_factory("gray", lambda: SolidColorStrategy((128, 128, 128)))
masking_registry.register_factory("white", lambda: SolidColorStrategy((255, 255, 255)))

from .blur import BlurStrategy  # noqa: F401 — 触发注册
from .mosaic import MosaicStrategy  # noqa: F401
