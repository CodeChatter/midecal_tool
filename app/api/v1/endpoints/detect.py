"""检测接口"""

import os
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ....core import cos
from ....schemas import DetectRequest, SensitiveItem, DetectResponse
from ....utils.image import validate_image_ext, validate_image_content
from ..deps import get_masker, get_semaphore, run_in_thread

router = APIRouter()
logger = logging.getLogger(__name__)


def _detect_sync(loc, req):
    """同步检测逻辑，在线程池中执行。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        suffix = Path(loc.key).suffix or ".jpg"
        input_path = os.path.join(tmp_dir, f"input{suffix}")
        cos.download_to_local(loc, input_path)
        validate_image_content(input_path)

        masker = get_masker(req.llm_provider)
        regions, stats = masker.detect_sensitive(input_path, req.categories)

        items = [
            SensitiveItem(
                text=r.text,
                category=r.category,
                bbox=r.bbox,
                ocr_line_index=r.ocr_line_index,
            )
            for r in regions
        ]

        return DetectResponse(
            sensitive_items=items,
            total_lines=stats.total_lines,
            sensitive_count=stats.sensitive_count,
        )


@router.post("/detect", response_model=DetectResponse)
async def detect_sensitive(req: DetectRequest):
    """接收图片 URL，仅检测敏感信息，返回 JSON（不打码）。"""
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
            return await run_in_thread(_detect_sync, loc, req)
        except Exception as e:
            logger.error(f"检测失败: {e}")
            raise HTTPException(500, f"检测失败: {e}")