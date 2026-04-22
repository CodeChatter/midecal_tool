"""打码接口"""

import os
import tempfile
import time
import logging
from pathlib import Path

from fastapi import APIRouter

from ....core.registry import masking_registry
from ....core import cos
from ....core.cos import CosLocation
from ....core.errors import SystemOverloadError
from ....schemas import MaskRequest, MaskResponse
from ....utils.image import validate_image_ext, validate_image_content
from ..deps import get_masker, get_semaphore, run_in_thread

router = APIRouter()
logger = logging.getLogger(__name__)


def _mask_sync(loc, req):
    """同步打码逻辑，在线程池中执行。"""
    import sys

    def _log(msg):
        logger.info(msg)
        print(f"[MASK] {msg}", file=sys.stderr, flush=True)

    t_total = time.time()
    _log(f"开始处理: bucket={loc.bucket}, key={loc.key}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. 从 COS 下载原图
        suffix = Path(loc.key).suffix or ".jpg"
        input_path = os.path.join(tmp_dir, f"input{suffix}")
        try:
            t0 = time.time()
            _log("[STEP 1/3] 开始从 COS 下载原图...")
            cos.download_to_local(loc, input_path)
            download_ms = (time.time() - t0) * 1000
            input_size_kb = os.path.getsize(input_path) // 1024 if os.path.exists(input_path) else 0
            _log(f"[STEP 1/3] COS 下载完成，耗时 {download_ms/1000:.1f}s，文件大小 {input_size_kb}KB")
            validate_image_content(input_path)
        except ValueError as e:
            _log(f"文件格式校验失败: {e}")
            return MaskResponse(success=False, origin_url=req.image_url, error=str(e))
        except Exception as e:
            logger.exception("图片下载失败")
            _log(f"图片下载失败: {e}")
            return MaskResponse(success=False, origin_url=req.image_url, error=f"图片下载失败: {e}")

        # 2. 打码处理（OCR + LLM 检测 + 遮罩）
        output_path = os.path.join(tmp_dir, f"output{suffix}")
        try:
            t0 = time.time()
            _log("[STEP 2/3] 开始脱敏处理...")
            masker = get_masker(req.llm_provider, req.mask_mode)
            masker.process_image(input_path, output_path, req.categories)
            process_ms = (time.time() - t0) * 1000
            output_exists = os.path.exists(output_path)
            output_size_kb = os.path.getsize(output_path) // 1024 if output_exists else 0
            _log(f"[STEP 2/3] 脱敏处理完成，耗时 {process_ms/1000:.1f}s，输出大小 {output_size_kb}KB")
        except SystemOverloadError as e:
            logger.exception("脱敏处理失败（系统资源不足）")
            _log(f"脱敏处理失败（系统资源不足）: {e}")
            return MaskResponse(
                success=False,
                origin_url=req.image_url,
                error="当前系统压力过大，请稍后重试",
            )
        except Exception as e:
            logger.exception("脱敏处理失败")
            _log(f"脱敏处理失败: {e}")
            return MaskResponse(
                success=False,
                origin_url=req.image_url,
                error="脱敏处理失败，请稍后重试",
            )

        if not os.path.exists(output_path):
            output_path = input_path

        # 3. 上传打码图片到新 URL
        try:
            t0 = time.time()
            _log("[STEP 3/3] 开始上传打码图片...")
            masked_key = cos.build_masked_key(loc.key)
            masked_loc = CosLocation(bucket=loc.bucket, region=loc.region, key=masked_key)
            masked_url = cos.upload_file(masked_loc, output_path)
            upload_ms = (time.time() - t0) * 1000
            upload_size_kb = os.path.getsize(output_path) // 1024 if os.path.exists(output_path) else 0
            _log(f"[STEP 3/3] 上传完成，耗时 {upload_ms/1000:.1f}s，上传大小 {upload_size_kb}KB")
        except Exception as e:
            logger.exception("上传打码图片失败")
            _log(f"上传打码图片失败: {e}")
            return MaskResponse(success=False, origin_url=req.image_url, error=f"上传打码图片失败: {e}")

    total_ms = (time.time() - t_total) * 1000
    _log(
        f"[DONE] 全流程完成，总耗时 {total_ms/1000:.1f}s "
        f"(download_ms={download_ms:.1f}, process_ms={process_ms:.1f}, upload_ms={upload_ms:.1f})"
    )
    return MaskResponse(
        success=True,
        origin_url=req.image_url,
        masked_url=masked_url,
    )


@router.post("/mask", response_model=MaskResponse)
async def mask_image(req: MaskRequest):
    """接收 COS 图片 URL，备份原图、打码、回传，返回 JSON。"""
    logger.info("[MASK] 收到 /api/mask 请求参数: %s", req.model_dump())
    if req.mask_mode not in masking_registry.available():
        logger.warning("[MASK] 无效的 mask_mode: %s", req.mask_mode)
        return MaskResponse(
            success=False,
            origin_url=req.image_url,
            error=f"无效的 mask_mode，可选: {masking_registry.available()}",
        )

    # 校验文件扩展名是否为图片
    try:
        validate_image_ext(req.image_url)
    except ValueError as e:
        logger.warning("[MASK] 图片扩展名校验失败: %s", e)
        return MaskResponse(success=False, origin_url=req.image_url, error=str(e))

    # 解析 URL 得到 bucket / region / key
    try:
        loc = cos.parse_url(req.image_url)
    except ValueError as e:
        logger.warning("[MASK] 图片 URL 解析失败: %s", e)
        return MaskResponse(success=False, origin_url=req.image_url, error=str(e))

    async with get_semaphore():
        logger.info(
            "[MASK] 请求进入线程池: bucket=%s, key=%s, provider=%s, mode=%s, categories=%s",
            loc.bucket,
            loc.key,
            req.llm_provider,
            req.mask_mode,
            req.categories,
        )
        try:
            resp = await run_in_thread(_mask_sync, loc, req)
        except Exception:
            logger.exception("[MASK] 线程池执行失败")
            raise

        logger.info(
            "[MASK] 请求处理完成: success=%s, masked_url=%s, error=%s",
            resp.success,
            resp.masked_url,
            resp.error,
        )
        return resp
