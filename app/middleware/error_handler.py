"""全局异常处理 + CORS + 请求日志"""

import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个请求的方法、路径、状态码和耗时。"""

    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.time()

        # 请求进入
        client = request.client.host if request.client else "-"
        logger.info(
            f"[{request_id}] --> {request.method} {request.url.path} "
            f"from {client}"
        )

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            logger.error(
                f"[{request_id}] <-- {request.method} {request.url.path} "
                f"500 UNHANDLED {elapsed:.0f}ms"
            )
            raise

        elapsed = (time.time() - start) * 1000
        logger.info(
            f"[{request_id}] <-- {request.method} {request.url.path} "
            f"{response.status_code} {elapsed:.0f}ms"
        )
        response.headers["X-Request-ID"] = request_id
        return response


def setup_middleware(app: FastAPI):
    """配置 CORS、请求日志和全局异常处理。"""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"未处理异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "内部服务器错误"},
        )
