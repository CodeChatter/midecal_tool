"""集中配置管理 — 从环境变量 / .env 读取"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_VISION_DETAIL: str = "auto"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_BASE_URL: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    ZHIPU_API_KEY: str = ""
    ZHIPU_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"
    ZHIPU_MODEL: str = "GLM-4.6V-Flash"

    BAILIAN_API_KEY: str = ""
    BAILIAN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    BAILIAN_MODEL: str = "qwen3-vl-32b-instruct"

    ARK_API_KEY: str = ""
    ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    ARK_MODEL: str = "doubao-seed-1-8-251228"
    ARK_MODEL_LITE: str = "doubao-seed-2-0-lite-260215"

    DEFAULT_LLM_PROVIDER: str = "volcengine-lite"

    COS_SECRET_ID: str = ""
    COS_SECRET_KEY: str = ""
    COS_REGION: str = "ap-nanjing"
    COS_BUCKET: str = ""
    COS_SCHEME: str = "https"
    # 自定义域名映射（JSON），格式：{"域名": "bucket/region", ...}
    # 示例：{"cdn.example.com": "example-1250000000/ap-nanjing"}
    COS_DOMAIN_MAP: str = ""

    MAX_CONCURRENCY: int = 2   # 最大并发处理数
    OCR_MODEL: str = "mobile"  # server（精度高）或 mobile（速度快）
    OCR_POOL_SIZE: int = 0     # OCR 实例池大小，0=自动（CPU核数-1）
    VISION_MAX_TOKENS: int = 384
    VISION_IMAGE_MAX_SIDE: int = 1280
    VISION_JPEG_QUALITY: int = 75

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
