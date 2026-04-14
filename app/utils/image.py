"""图像工具函数"""

import base64
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from ..core.models import TextLine

logger = logging.getLogger(__name__)


# 支持的图片扩展名
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

# 图片文件头魔数 → 格式名
_MAGIC_BYTES = [
    (b'\xff\xd8\xff', 'JPEG'),
    (b'\x89PNG\r\n\x1a\n', 'PNG'),
    (b'BM', 'BMP'),
    (b'RIFF', 'WEBP'),  # WEBP 以 RIFF 开头
    (b'II\x2a\x00', 'TIFF'),  # little-endian
    (b'MM\x00\x2a', 'TIFF'),  # big-endian
]


def validate_image_ext(url_or_path: str) -> None:
    """校验 URL 或文件路径的扩展名是否为支持的图片格式。"""
    from pathlib import Path
    ext = Path(url_or_path.split('?')[0]).suffix.lower()
    if ext and ext not in ALLOWED_IMAGE_EXTS:
        raise ValueError(
            f"不支持的文件格式 '{ext}'，仅支持图片: {', '.join(sorted(ALLOWED_IMAGE_EXTS))}"
        )


def validate_image_content(file_path: str) -> None:
    """读取文件头部字节，校验是否为真实图片文件。"""
    with open(file_path, 'rb') as f:
        header = f.read(16)

    if len(header) < 2:
        raise ValueError("文件为空或过小，无法识别为图片")

    for magic, fmt in _MAGIC_BYTES:
        if header.startswith(magic):
            return  # 匹配到已知图片格式

    raise ValueError(
        f"文件内容不是有效图片（文件头: {header[:8].hex()}），"
        f"仅支持 JPEG/PNG/BMP/WEBP/TIFF 格式"
    )


def image_to_base64(image_path: str,
                    max_side: int = 2048,
                    jpeg_quality: int = 85) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
    """读取图片文件并返回 (base64_data, media_type, original_size, sent_size)。

    仅对超大图（最长边 > max_side）做缩放压缩以减少 LLM 传输体积。
    小图或模糊图保持原样，避免进一步损失清晰度。
    """
    import cv2

    img = cv2.imread(image_path)
    if img is not None:
        h, w = img.shape[:2]
        original_size = (w, h)
        max_dim = max(h, w)
        if max_dim > max_side:
            scale = max_side / max_dim
            sent_w, sent_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (sent_w, sent_h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', img,
                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise ValueError(f"图片 JPEG 编码失败: {image_path}")
            data = base64.b64encode(buf.tobytes()).decode('utf-8')
            logger.info(
                "Vision 图片压缩: path=%s, original=%sx%s, sent=%sx%s, max_side=%s, jpeg_quality=%s, payload_kb=%s",
                image_path, w, h, sent_w, sent_h, max_side, jpeg_quality, len(data) // 1024,
            )
            return data, 'image/jpeg', original_size, (sent_w, sent_h)

        with open(image_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        logger.info(
            "Vision 图片直传: path=%s, original=%sx%s, payload_kb=%s",
            image_path, w, h, len(data) // 1024,
        )
        return data, _media_type_from_ext(image_path), original_size, original_size

    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    media_type = _media_type_from_ext(image_path)
    logger.info(
        "Vision 图片退回原文件读取: path=%s, payload_kb=%s, media_type=%s",
        image_path, len(data) // 1024, media_type,
    )
    return data, media_type, (0, 0), (0, 0)


def _media_type_from_ext(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.bmp': 'image/bmp',
        '.webp': 'image/webp', '.tiff': 'image/tiff',
    }
    return media_type_map.get(ext, 'image/jpeg')


def get_value_polygon(line: TextLine, value: str,
                      padding: int = 2,
                      prefix_miss: int = 0) -> np.ndarray:
    """
    根据文本行中的子串位置，计算对应的四边形多边形坐标。

    prefix_miss: OCR 漏识的前缀字符数（如 "李正国" OCR 只识别出 "正国"，
                 则 prefix_miss=1），将 start_ratio 向左扩展相应宽度以覆盖漏字。
    """
    import logging
    _logger = logging.getLogger(__name__)

    text = line.text
    raw_pts = np.array(line.bbox, dtype=np.float64)

    # 验证 bbox 至少有 4 个点、每点有 2 个坐标，否则用外接矩形兜底
    if raw_pts.ndim != 2 or raw_pts.shape[0] < 4 or raw_pts.shape[1] < 2:
        x1, y1, x2, y2 = line.xyxy
        raw_pts = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64
        )
        _logger.debug(f"OCR bbox 形状异常（{line.bbox}），降级为外接矩形")

    pts = raw_pts
    line_h = max(line.xyxy[3] - line.xyxy[1], 1)
    pad = max(padding, int(line_h * 0.15))

    pos = text.find(value)
    if pos < 0:
        text_clean = text.replace(' ', '')
        value_clean = value.replace(' ', '')
        pos_clean = text_clean.find(value_clean)
        if pos_clean >= 0:
            count = 0
            for i, ch in enumerate(text):
                if ch != ' ':
                    if count == pos_clean:
                        pos = i
                        break
                    count += 1

    total_len = max(len(text), 1)
    if pos < 0 or (pos == 0 and len(value) >= total_len):
        start_ratio, end_ratio = 0.0, 1.0
    else:
        end_ratio = (pos + len(value)) / total_len
        char_w = 1.0 / total_len
        start_ratio = max(0.0, pos / total_len - prefix_miss * char_w)

    p_tl, p_tr, p_br, p_bl = pts[0], pts[1], pts[2], pts[3]
    top_vec = p_tr - p_tl
    bottom_vec = p_br - p_bl

    new_tl = p_tl + top_vec * start_ratio
    new_tr = p_tl + top_vec * end_ratio
    new_br = p_bl + bottom_vec * end_ratio
    new_bl = p_bl + bottom_vec * start_ratio

    poly = np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float64)

    center = poly.mean(axis=0)
    expanded = []
    for pt in poly:
        direction = pt - center
        norm = np.linalg.norm(direction)
        if norm > 0:
            expanded.append(pt + direction / norm * pad)
        else:
            expanded.append(pt)

    return np.array(expanded, dtype=np.int32)

