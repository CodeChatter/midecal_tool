"""遮罩策略抽象基类"""

from abc import ABC, abstractmethod

import numpy as np


class BaseMaskStrategy(ABC):
    """所有遮罩策略必须实现此接口。"""

    @abstractmethod
    def apply(self, image: np.ndarray, polygon: np.ndarray) -> None:
        """在 image 上对 polygon 区域施加遮罩（原地修改）。"""
        ...
