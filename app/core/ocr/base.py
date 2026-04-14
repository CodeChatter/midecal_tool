"""OCR 引擎抽象基类"""

from abc import ABC, abstractmethod
from typing import List

from ..models import TextLine


class BaseOCREngine(ABC):
    """所有 OCR 引擎必须实现此接口。"""

    @abstractmethod
    def recognize(self, image_path: str) -> List[TextLine]:
        """识别图片中的文本行，返回 TextLine 列表。"""
        ...
