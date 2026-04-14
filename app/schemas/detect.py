from typing import List, Optional, Tuple

from pydantic import BaseModel


class DetectRequest(BaseModel):
    image_url: str
    llm_provider: Optional[str] = None
    categories: Optional[List[str]] = None


class SensitiveItem(BaseModel):
    text: str
    category: str
    bbox: Tuple[int, int, int, int]
    ocr_line_index: int


class DetectResponse(BaseModel):
    sensitive_items: List[SensitiveItem]
    total_lines: int
    sensitive_count: int
