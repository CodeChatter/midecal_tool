"""腾讯云 COS 客户端封装"""

import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Dict, NamedTuple
import json
from urllib.parse import unquote, urlparse

from qcloud_cos import CosConfig, CosS3Client

from ..config import get_settings

logger = logging.getLogger(__name__)

# 按 region 缓存 client（不同 region 需要不同 client）
_clients: Dict[str, CosS3Client] = {}


class CosLocation(NamedTuple):
    """从 URL 解析出的 COS 定位信息。"""
    bucket: str
    region: str
    key: str


_domain_map_cache: dict = {}


def _get_domain_map() -> dict:
    """解析 COS_DOMAIN_MAP 配置，返回 {域名: (bucket, region)} 字典。"""
    if _domain_map_cache:
        return _domain_map_cache
    s = get_settings()
    if not s.COS_DOMAIN_MAP:
        return {}
    try:
        raw = json.loads(s.COS_DOMAIN_MAP)
        for domain, value in raw.items():
            bucket, region = value.split("/", 1)
            _domain_map_cache[domain] = (bucket.strip(), region.strip())
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"COS_DOMAIN_MAP 配置格式错误: {e}")
    return _domain_map_cache


def parse_url(url: str) -> CosLocation:
    """从 COS URL 解析 bucket、region、key。

    支持两种格式：
    1. 标准格式：https://<bucket>.cos.<region>.myqcloud.com/<key>
    2. 自定义域名：https://<custom-domain>/<key>
       需在 COS_DOMAIN_MAP 中配置域名映射
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    key = unquote(parsed.path.lstrip("/"))

    # 尝试标准 COS URL 格式（兼容 myqcloud.com 和 tencentcos.cn）
    match = re.match(r"^(.+)\.cos\.([^.]+)\.(myqcloud\.com|tencentcos\.cn)$", host)
    if match:
        return CosLocation(bucket=match.group(1), region=match.group(2), key=key)

    # 尝试自定义域名映射
    domain_map = _get_domain_map()
    if host in domain_map:
        bucket, region = domain_map[host]
        return CosLocation(bucket=bucket, region=region, key=key)

    raise ValueError(
        f"无法解析 COS URL: {url}。"
        f"支持标准格式 <bucket>.cos.<region>.myqcloud.com 或 "
        f"自定义域名（需在 COS_DOMAIN_MAP 中配置）"
    )


def _get_client(region: str) -> CosS3Client:
    if region not in _clients:
        s = get_settings()
        config = CosConfig(
            Region=region,
            SecretId=s.COS_SECRET_ID,
            SecretKey=s.COS_SECRET_KEY,
            Scheme=s.COS_SCHEME,
            PoolConnections=150,
            PoolMaxSize=150,
        )
        _clients[region] = CosS3Client(config)
    return _clients[region]


def build_backup_key(key: str) -> str:
    """构造备份 key：dir/stem_bak_时间戳.ext"""
    p = Path(key)
    ts = int(time.time())
    return str(p.with_name(f"{p.stem}_bak_{ts}{p.suffix}"))


def build_masked_key(key: str) -> str:
    """构造打码 key：dir/stem_masked_时间戳.ext"""
    p = Path(key)
    ts = int(time.time())
    return str(p.with_name(f"{p.stem}_masked_{ts}{p.suffix}"))


def build_url(bucket: str, region: str, key: str) -> str:
    """拼完整公网 URL（优先使用自定义域名）。"""
    s = get_settings()
    # 反查自定义域名
    for domain, (b, r) in _get_domain_map().items():
        if b == bucket and r == region:
            return f"{s.COS_SCHEME}://{domain}/{key}"
    return f"{s.COS_SCHEME}://{bucket}.cos.{region}.myqcloud.com/{key}"


def download_to_local(loc: CosLocation, local_path: str) -> str:
    """从 COS 下载对象到本地文件，返回本地路径。"""
    client = _get_client(loc.region)
    client.download_file(
        Bucket=loc.bucket,
        Key=loc.key,
        DestFilePath=local_path,
    )
    logger.info(f"下载成功: {loc.key} -> {local_path}")
    return local_path


def upload_file(loc: CosLocation, local_path: str) -> str:
    """上传本地文件到指定位置，返回完整 URL。"""
    client = _get_client(loc.region)

    content_type, _ = mimetypes.guess_type(local_path)
    if content_type is None:
        content_type = "application/octet-stream"

    client.upload_file(
        Bucket=loc.bucket,
        LocalFilePath=local_path,
        Key=loc.key,
        ContentType=content_type,
    )
    logger.info(f"上传成功: {loc.key}")
    return build_url(loc.bucket, loc.region, loc.key)


def generate_presigned_url(loc: CosLocation, expires: int = 600) -> str:
    """生成临时预签名下载 URL。

    Args:
        loc: COS 定位信息
        expires: 有效期（秒），默认 10 分钟
    """
    client = _get_client(loc.region)
    url = client.get_presigned_url(
        Method="GET",
        Bucket=loc.bucket,
        Key=loc.key,
        Expired=expires,
    )
    logger.info(f"生成预签名 URL: {loc.key}, 有效期 {expires}s")
    return url
