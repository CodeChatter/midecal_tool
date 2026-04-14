"""
pytest 配置：加载 .env，供 pytest 运行时使用。
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")