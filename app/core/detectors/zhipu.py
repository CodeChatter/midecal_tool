"""智谱 GLM Vision 检测器（OpenAI 兼容接口）"""

import logging

import httpx

from ...config import get_settings
from ..registry import detector_registry
from .base import BaseDetector

logger = logging.getLogger(__name__)


@detector_registry.register("zhipu")
class ZhipuDetector(BaseDetector):

    def __init__(self):
        settings = get_settings()
        api_key = settings.ZHIPU_API_KEY
        if not api_key:
            raise RuntimeError("ZHIPU_API_KEY 未配置")
        from openai import OpenAI
        timeout = httpx.Timeout(timeout=120.0, connect=10.0)
        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.ZHIPU_BASE_URL,
            timeout=timeout,
        )
        self._model = settings.ZHIPU_MODEL
        self._max_tokens = settings.VISION_MAX_TOKENS
        logger.info(f"Vision LLM 就绪：智谱 ({self._model}, max_tokens={self._max_tokens})")

    def _call_vision(self, b64_data: str, media_type: str,
                     user_prompt: str, system_prompt: str) -> str:
        logger.info(f"调用智谱 Vision API: model={self._model}, max_tokens={self._max_tokens}, image_size={len(b64_data)//1024}KB")
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
        logger.info("智谱 Vision API 调用完成")
        return resp.choices[0].message.content
