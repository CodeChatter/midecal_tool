"""依赖注入工厂 — 组件创建与缓存"""

import asyncio
import fcntl
import os
import logging
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx

from ...config import get_settings
from ...core.registry import ocr_registry, detector_registry, masking_registry

# 确保各模块的注册装饰器被执行
import app.core.ocr  # noqa: F401
import app.core.detectors  # noqa: F401
import app.core.masking  # noqa: F401

from ...core.pipeline import MedicalPrivacyMasker
from ...core.ocr.base import BaseOCREngine

logger = logging.getLogger(__name__)

# ── 并发控制 ──
_semaphore: Optional[asyncio.Semaphore] = None
_thread_pool: Optional[ThreadPoolExecutor] = None
_PREWARM_LOCK_FILE = Path("/tmp/medical_tool_ocr_prewarm.lock")


@contextmanager
def _prewarm_file_lock():
    _PREWARM_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _PREWARM_LOCK_FILE.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def get_semaphore() -> asyncio.Semaphore:
    """获取全局并发信号量（懒初始化，首次调用时根据配置创建）。"""
    global _semaphore
    if _semaphore is None:
        limit = get_settings().MAX_CONCURRENCY
        _semaphore = asyncio.Semaphore(limit)
        logger.info(f"并发控制初始化: MAX_CONCURRENCY={limit}")
    return _semaphore


def _get_thread_pool() -> ThreadPoolExecutor:
    """获取全局线程池（大小与 MAX_CONCURRENCY 一致）。"""
    global _thread_pool
    if _thread_pool is None:
        limit = get_settings().MAX_CONCURRENCY
        _thread_pool = ThreadPoolExecutor(max_workers=limit, thread_name_prefix="worker")
        logger.info(f"线程池初始化: max_workers={limit}")
    return _thread_pool


async def run_in_thread(func, *args, **kwargs):
    """将同步阻塞函数放到线程池中执行，避免阻塞事件循环。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_thread_pool(),
        partial(func, *args, **kwargs),
    )

# 单例缓存
_ocr_engines: Dict[str, BaseOCREngine] = {}
_maskers: Dict[str, MedicalPrivacyMasker] = {}


def log_runtime_settings() -> None:
    settings = get_settings()
    env_path = Path.cwd() / ".env"
    paddle_device = "unavailable"
    paddle_cuda = "unknown"
    install_gpu = os.environ.get("INSTALL_GPU", "")
    app_workers = os.environ.get("APP_WORKERS", "1")
    try:
        import paddle

        paddle_cuda = str(paddle.device.is_compiled_with_cuda())
        paddle_device = paddle.device.get_device()
    except Exception as exc:
        paddle_device = f"error:{exc}"

    logger.info(
        "运行配置: cwd=%s, env_file_exists=%s, INSTALL_GPU=%s, APP_WORKERS=%s, MAX_CONCURRENCY=%s, OCR_POOL_SIZE=%s, OCR_MODEL=%s, DEFAULT_LLM_PROVIDER=%s, paddle_cuda=%s, paddle_device=%s",
        Path.cwd(),
        env_path.exists(),
        install_gpu,
        app_workers,
        settings.MAX_CONCURRENCY,
        settings.OCR_POOL_SIZE,
        settings.OCR_MODEL,
        settings.DEFAULT_LLM_PROVIDER,
        paddle_cuda,
        paddle_device,
    )
    if install_gpu.lower() == "true" and (paddle_cuda != "True" or paddle_device.startswith("cpu")):
        logger.warning("GPU 模式已启用，但当前 Paddle 设备不是 GPU: paddle_cuda=%s, paddle_device=%s", paddle_cuda, paddle_device)


def prewarm_ocr_engine(name: str = "paddle") -> BaseOCREngine:
    if name in _ocr_engines:
        logger.info(f"OCR 引擎已就绪，跳过重复预热: {name}")
        return _ocr_engines[name]

    logger.info(f"等待 OCR 引擎预热锁: {name}")
    with _prewarm_file_lock():
        if name in _ocr_engines:
            logger.info(f"OCR 引擎已由当前进程初始化: {name}")
            return _ocr_engines[name]
        engine = get_ocr_engine(name)
        # 用小白图做一次真正推理，触发 PaddlePaddle C++ 运行时初始化，
        # 避免首次请求时因底层 SIGSEGV 导致 worker 进程崩溃。
        if hasattr(engine, 'warmup'):
            engine.warmup()
        logger.info(f"OCR 引擎预热完成: {name}")
        return engine


def get_masker(provider: Optional[str] = None,
               mode: str = "blur") -> MedicalPrivacyMasker:
    """获取或创建 MedicalPrivacyMasker 实例（按 provider:mode 缓存）。"""
    provider = provider or get_settings().DEFAULT_LLM_PROVIDER
    key = f"{provider}:{mode}"
    if key not in _maskers:
        # OCR 引擎单例
        ocr_key = "paddle"
        if ocr_key not in _ocr_engines:
            _ocr_engines[ocr_key] = ocr_registry.create(ocr_key)
        ocr = _ocr_engines[ocr_key]

        detector = detector_registry.create(provider)
        mask_strategy = masking_registry.create(mode)

        _maskers[key] = MedicalPrivacyMasker(
            ocr=ocr,
            detector=detector,
            mask_strategy=mask_strategy,
        )
        logger.info(f"创建 masker: provider={provider}, mode={mode}")
    return _maskers[key]


def get_ocr_engine(name: str = "paddle") -> BaseOCREngine:
    """获取或创建 OCR 引擎实例（单例缓存）。"""
    if name not in _ocr_engines:
        _ocr_engines[name] = ocr_registry.create(name)
        logger.info(f"创建 OCR 引擎: {name}")
    return _ocr_engines[name]


def get_maskers() -> Dict[str, MedicalPrivacyMasker]:
    """供 lifespan 访问的 masker 字典。"""
    return _maskers


# Content-Type → 扩展名 映射
_MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/tiff": ".tiff",
}


def download_image(url: str, tmp_dir: str) -> str:
    """从 URL 下载图片到临时目录，返回本地文件路径。"""
    # 先尝试从 URL 路径推断扩展名
    parsed = urlparse(url)
    url_path = parsed.path
    ext = Path(url_path).suffix.lower() if url_path else ""

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    # 如果 URL 没有有效扩展名，从 Content-Type 推断
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"):
        content_type = resp.headers.get("content-type", "")
        # 取 mime 主类型（去掉参数如 charset）
        mime = content_type.split(";")[0].strip().lower()
        ext = _MIME_TO_EXT.get(mime, ".jpg")

    tmp_path = os.path.join(tmp_dir, f"input{ext}")
    with open(tmp_path, "wb") as f:
        f.write(resp.content)

    return tmp_path
