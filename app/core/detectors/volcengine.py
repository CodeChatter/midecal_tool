"""火山引擎豆包 Vision 检测器（OpenAI 兼容接口）"""

import logging

import httpx

from ...config import get_settings
from ..registry import detector_registry
from .base import BaseDetector

logger = logging.getLogger(__name__)


class _VolcEngineDetectorBase(BaseDetector):
    """火山引擎检测器基类，子类通过 _get_model() 指定模型。"""

    def _get_model(self, settings) -> str:
        raise NotImplementedError

    def __init__(self):
        settings = get_settings()
        api_key = settings.ARK_API_KEY
        if not api_key:
            raise RuntimeError("ARK_API_KEY 未配置")
        from openai import OpenAI
        concurrency = settings.MAX_CONCURRENCY
        timeout = httpx.Timeout(timeout=300.0, connect=15.0)
        transport = httpx.HTTPTransport(
            limits=httpx.Limits(
                max_connections=concurrency + 20,
                max_keepalive_connections=concurrency,
            ),
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.ARK_BASE_URL,
            timeout=timeout,
            max_retries=0,
            http_client=httpx.Client(transport=transport, timeout=timeout),
        )
        self._model = self._get_model(settings)
        self._max_tokens = settings.VISION_MAX_TOKENS
        logger.info(f"Vision LLM 就绪：火山引擎 ({self._model}, pool={concurrency}, max_tokens={self._max_tokens})")

    def _call_vision(self, b64_data: str, media_type: str,
                     user_prompt: str, system_prompt: str) -> str:
        logger.info(f"调用火山引擎 Vision API: model={self._model}, max_tokens={self._max_tokens}, image_size={len(b64_data)//1024}KB")
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
                        }
                    },
                    {'type': 'text', 'text': user_prompt},
                ]}
            ]
        )
        logger.info("火山引擎 Vision API 调用完成")
        return resp.choices[0].message.content


@detector_registry.register("volcengine")
class VolcEngineDetector(_VolcEngineDetectorBase):
    def _get_model(self, settings) -> str:
        return settings.ARK_MODEL


@detector_registry.register("volcengine-lite")
class VolcEngineDetectorLite(_VolcEngineDetectorBase):
    def _get_model(self, settings) -> str:
        return settings.ARK_MODEL_LITE