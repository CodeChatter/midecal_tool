"""OpenAI Vision 检测器"""

import logging

import httpx

from ...config import get_settings
from ..registry import detector_registry
from .base import BaseDetector

logger = logging.getLogger(__name__)


@detector_registry.register("openai")
class OpenAIDetector(BaseDetector):

    def __init__(self):
        settings = get_settings()
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置")
        from openai import OpenAI
        base_url = settings.OPENAI_BASE_URL or None
        concurrency = settings.MAX_CONCURRENCY
        timeout = httpx.Timeout(timeout=300.0, connect=15.0)
        transport = httpx.HTTPTransport(
            limits=httpx.Limits(
                max_connections=concurrency + 20,
                max_keepalive_connections=concurrency,
            ),
        )
        http_client = httpx.Client(transport=transport, timeout=timeout)
        self._client = (
            OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
                http_client=http_client,
            )
            if base_url else
            OpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=0,
                http_client=http_client,
            )
        )
        self._model = settings.OPENAI_MODEL
        self._max_tokens = settings.VISION_MAX_TOKENS
        detail = (settings.OPENAI_VISION_DETAIL or "auto").lower()
        if detail not in {"auto", "low", "high"}:
            detail = "auto"
        self._detail = detail
        logger.info(
            f"Vision LLM 就绪：OpenAI ({self._model}, pool={concurrency}, max_tokens={self._max_tokens}, detail={self._detail})"
        )

    def _call_vision(self, b64_data: str, media_type: str,
                     user_prompt: str, system_prompt: str) -> str:
        logger.info(
            f"调用 OpenAI Vision API: model={self._model}, max_tokens={self._max_tokens}, detail={self._detail}, image_size={len(b64_data)//1024}KB"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            max_tokens=self._max_tokens,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:{media_type};base64,{b64_data}',
                            'detail': self._detail,
                        }
                    },
                    {'type': 'text', 'text': user_prompt},
                ]}
            ]
        )
        logger.info("OpenAI Vision API 调用完成")
        return resp.choices[0].message.content
