from typing import List, Tuple

from pydantic import BaseModel


class OcrRequest(BaseModel):
    image_url: str


class OcrTextLine(BaseModel):
    index: int
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float


class OcrResponse(BaseModel):
    lines: List[OcrTextLine]
    full_text: str
    total_lines: int