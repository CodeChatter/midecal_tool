from typing import List, Optional

from pydantic import BaseModel


class MaskRequest(BaseModel):
    image_url: str
    llm_provider: Optional[str] = None
    mask_mode: str = "white"
    categories: Optional[List[str]] = None


class MaskResponse(BaseModel):
    success: bool
    origin_url: Optional[str] = None
    masked_url: Optional[str] = None
    error: Optional[str] = None
