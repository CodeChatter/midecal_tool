"""纯色填充遮罩策略"""

from typing import Tuple

import cv2
import numpy as np

from .base import BaseMaskStrategy


class SolidColorStrategy(BaseMaskStrategy):
    """参数化纯色填充遮罩。"""

    def __init__(self, color: Tuple[int, int, int] = (0, 0, 0)):
        self.color = color

    def apply(self, image: np.ndarray, polygon: np.ndarray) -> None:
        pts = polygon.reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], self.color)
