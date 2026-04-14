"""OCR 文字提取接口"""

import os
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ....core import cos
from ....schemas.ocr import OcrRequest, OcrTextLine, OcrResponse
from ....utils.image import validate_image_ext, validate_image_content
from ..deps import get_ocr_engine, get_semaphore, run_in_thread

router = APIRouter()
logger = logging.getLogger(__name__)


def _ocr_sync(loc):
    """同步 OCR 逻辑，在线程池中执行。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        suffix = Path(loc.key).suffix or ".jpg"
        input_path = os.path.join(tmp_dir, f"input{suffix}")
        cos.download_to_local(loc, input_path)
        validate_image_content(input_path)

        ocr = get_ocr_engine()
        text_lines = ocr.recognize(input_path)

        lines = [
            OcrTextLine(
                index=line.index,
                text=line.text,
                bbox=line.xyxy,
                confidence=line.confidence,
            )
            for line in text_lines
        ]

        full_text = "\n".join(line.text for line in text_lines)

        return OcrResponse(
            lines=lines,
            full_text=full_text,
            total_lines=len(lines),
        )


@router.post("/ocr", response_model=OcrResponse)
async def extract_text(req: OcrRequest):
    """接收图片 URL，提取全部文字内容并返回。"""
    try:
        validate_image_ext(req.image_url)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        loc = cos.parse_url(req.image_url)
    except ValueError as e:
        raise HTTPException(400, str(e))

    async with get_semaphore():
        try:
            return await run_in_thread(_ocr_sync, loc)
        except Exception as e:
            logger.error(f"OCR 识别失败: {e}")
            raise HTTPException(500, f"OCR 识别失败: {e}")