import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings


def test_get_settings_reads_effective_environment(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("MAX_CONCURRENCY", "12")
    monkeypatch.setenv("OCR_POOL_SIZE", "0")
    monkeypatch.setenv("OCR_MODEL", "server")
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "volcengine")
    monkeypatch.setenv("VISION_MAX_TOKENS", "256")
    monkeypatch.setenv("VISION_IMAGE_MAX_SIDE", "960")
    monkeypatch.setenv("VISION_JPEG_QUALITY", "70")
    monkeypatch.setenv("OPENAI_VISION_DETAIL", "low")

    settings = get_settings()

    assert settings.MAX_CONCURRENCY == 12
    assert settings.OCR_POOL_SIZE == 0
    assert settings.OCR_MODEL == "server"
    assert settings.DEFAULT_LLM_PROVIDER == "volcengine"
    assert settings.VISION_MAX_TOKENS == 256
    assert settings.VISION_IMAGE_MAX_SIDE == 960
    assert settings.VISION_JPEG_QUALITY == 70
    assert settings.OPENAI_VISION_DETAIL == "low"

    get_settings.cache_clear()


def test_get_settings_reloads_after_cache_clear(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("MAX_CONCURRENCY", "8")
    first = get_settings()
    assert first.MAX_CONCURRENCY == 8

    get_settings.cache_clear()
    monkeypatch.setenv("MAX_CONCURRENCY", "16")
    second = get_settings()
    assert second.MAX_CONCURRENCY == 16

    get_settings.cache_clear()
