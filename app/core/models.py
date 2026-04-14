"""数据结构：TextLine, SensitiveRegion, ProcessStats"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class TextLine:
    """OCR 文本行"""
    index: int
    text: str
    bbox: List[List[int]]   # [[x,y], ...] 四点坐标
    confidence: float

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        pts = np.array(self.bbox)
        return (int(pts[:, 0].min()), int(pts[:, 1].min()),
                int(pts[:, 0].max()), int(pts[:, 1].max()))


@dataclass
class SensitiveRegion:
    """一处需要遮罩的敏感区域"""
    text: str
    category: str
    polygon: np.ndarray  # shape (4, 2) 四点多边形，跟随文字倾斜
    ocr_line_index: int = -1

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """轴对齐外接矩形"""
        return (int(self.polygon[:, 0].min()), int(self.polygon[:, 1].min()),
                int(self.polygon[:, 0].max()), int(self.polygon[:, 1].max()))


@dataclass
class ProcessStats:
    total_lines: int = 0
    sensitive_count: int = 0
    masked_count: int = 0
    orientation_ms: float = 0.0
    ocr_ms: float = 0.0
    rotation_retry_ms: float = 0.0
    detect_ms: float = 0.0
    mask_ms: float = 0.0
    pipeline_ms: float = 0.0
    sensitive_items: List[str] = field(default_factory=list)
