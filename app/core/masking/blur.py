"""高斯模糊遮罩策略"""

import cv2
import numpy as np

from ..registry import masking_registry
from .base import BaseMaskStrategy


@masking_registry.register("blur")
class BlurStrategy(BaseMaskStrategy):

    def apply(self, image: np.ndarray, polygon: np.ndarray) -> None:
        x1, y1, x2, y2 = (
            int(polygon[:, 0].min()), int(polygon[:, 1].min()),
            int(polygon[:, 0].max()), int(polygon[:, 1].max()),
        )
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 > x1 and y2 > y1:
            roi = image[y1:y2, x1:x2]
            h, w = roi.shape[:2]
            ksize = max(35, (min(h, w) // 2) | 1)
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            roi_mask = mask[y1:y2, x1:x2]
            roi[roi_mask > 0] = blurred[roi_mask > 0]
