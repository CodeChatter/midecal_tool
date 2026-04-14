"""Claude Vision 检测器"""

import logging

import httpx

from ...config import get_settings
from ..registry import detector_registry
from .base import BaseDetector

logger = logging.getLogger(__name__)


@detector_registry.register("claude")
class ClaudeDetector(BaseDetector):

    def __init__(self):
        settings = get_settings()
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY 未配置")
        import anthropic
        timeout = httpx.Timeout(timeout=120.0, connect=10.0)
        kwargs = {'api_key': api_key, 'timeout': timeout}
        base_url = settings.ANTHROPIC_BASE_URL or None
        if base_url:
            kwargs['base_url'] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        self._model = settings.ANTHROPIC_MODEL
        self._max_tokens = settings.VISION_MAX_TOKENS
        logger.info(f"Vision LLM 就绪：Claude ({self._model}, max_tokens={self._max_tokens})")

    def _call_vision(self, b64_data: str, media_type: str,
                     user_prompt: str, system_prompt: str) -> str:
        logger.info(f"调用 Claude Vision API: model={self._model}, max_tokens={self._max_tokens}, image_size={len(b64_data)//1024}KB")
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': media_type,
                            'data': b64_data,
                        }
                    },
                    {'type': 'text', 'text': user_prompt},
                ]
            }]
        )
        logger.info("Claude Vision API 调用完成")
        return resp.content[0].text
