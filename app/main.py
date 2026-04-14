"""FastAPI 应用创建 + lifespan"""

import logging
from contextlib import asynccontextmanager

from .bootstrap.runtime import bootstrap_runtime

# 生产 ASGI 入口：必须先完成运行时初始化，再导入应用装配链路。
bootstrap_runtime()

from fastapi import FastAPI

from .config import get_settings
from .api.v1 import v1_router
from .api.v1.deps import get_masker, get_maskers, log_runtime_settings, prewarm_ocr_engine
from .middleware.error_handler import setup_middleware

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log_runtime_settings()
    try:
        prewarm_ocr_engine()
        get_masker(settings.DEFAULT_LLM_PROVIDER)
        logger.info("默认 masker 预加载完成")
    except Exception as e:
        logger.warning(f"预加载 masker 失败（将在首次请求时重试）: {e}")
    yield
    get_maskers().clear()


app = FastAPI(
    title="医疗单据隐私脱敏 API",
    description="OCR + Vision LLM 混合脱敏服务",
    version="1.0.0",
    lifespan=lifespan,
)

setup_middleware(app)

# 顶级健康检查
@app.get("/health")
async def health():
    return {"status": "ok"}

# v1 路由（/api/mask, /api/detect）
app.include_router(v1_router)
