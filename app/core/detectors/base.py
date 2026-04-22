"""Vision LLM 检测器抽象基类 + 共享逻辑"""

import re
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..models import TextLine, SensitiveRegion
from ...utils.image import image_to_base64, get_value_polygon

logger = logging.getLogger(__name__)


def _perf(msg: str):
    logger.info(msg)
    print(f"[PERF] {msg}", file=sys.stderr, flush=True)

# ============================================================
# 默认敏感类别
# ============================================================

DEFAULT_CATEGORIES = [
    "name", "id_card", "phone", "medical_id", "doctor",
    "email", "address", "hospital", "portrait",
]

CATEGORY_DESCRIPTIONS = {
    "name": ("所有真实姓名中的非医务人员姓名，包括患者、就诊人、受检者、家属、联系人、"
             "采样人、送检人、录入人等；如“姓名：”“患者姓名：”“受检者：”等字段中的"
             "实际姓名值（只返回姓名，不返回标签）。医务人员姓名请归类为 doctor"),
    "id_card": "身份证号码（15或18位数字）",
    "phone": "手机或固定电话号码",
    "medical_id": "病历号/就诊号/住院号/超声号/门诊号/检查号等各类编号",
    "email": "电子邮箱地址（如 xxx@xxx.com）",
    "address": "家庭住址/居住地址/户籍地址/通讯地址等个人地址信息",
    "age": '年龄数值（如"33岁"）',
    "date_of_birth": "出生日期",
    "date": ("所有日期格式的内容，包括检查日期、报告日期、打印日期、入院日期、出院日期、"
             "手术日期等，常见格式如 2026-01-18、2026/01/18、2026.01.18、2026年1月18日"),
    "hospital": ("医院全称或简称，包括附属医院、分院、院区、诊所等机构名称；"
                 "单据顶部医院标题也属于 hospital，如“无锡市第二人民医院南院”"),
    "doctor": ("所有出现在医疗单据上的医生/医务人员真实姓名，包括申请医生、报告医生、"
               "审核医师、检查医生、主治医师、经治医生、操作者、技师、签名医生等"),
    "portrait": ("图片中可见的真人人像照片区域，包括证件照、头像、脸部特写、半身照等；"
                 "仅限真实人像，不含卡通、插画、解剖示意图、医学影像（如 CT/X 光/B 超）"),
}

# ============================================================
# Prompt 构建
# ============================================================

_PROMPT_HEADER = """你是医疗文档隐私脱敏专家。

你会收到：
1. 一张医疗单据的图片
2. OCR 识别出的所有文本行（JSON 数组，每行有 index 和 text）

你的任务：对照图片和 OCR 文本，找出所有包含个人隐私信息的文本行。

⚠️ 关键要求：
- 图片中可能有多张文档叠放在一起，你必须逐一检查每一张可见文档
- 严格对照图片内容，不要虚构 OCR 中不存在的文本
- 宁可多标不可漏标
- 所有真实姓名都属于敏感信息，不能遗漏
- `name` 类包括所有非医务人员姓名，如“姓名 / 患者姓名 / 受检者 / 就诊人 / 联系人 / 家属 / 采样人 / 送检人”等字段后的实际姓名值
- `doctor` 类包括所有医生/医务人员姓名，如“申请医生 / 报告医生 / 审核医师 / 检查医生 / 主治医师 / 经治医生 / 操作者 / 技师”等字段后的实际姓名值
- 医生、医师、技师、检查者、审核者等医务人员姓名必须归类为 `doctor`，不要归到 `name`
- `hospital` 类包括医院全称、简称、院区、分院名称，以及单据顶部医院标题
- 当请求包含 `date` 类时，所有日期格式的内容都属于敏感信息，包括检查日期、报告日期、打印日期、入院日期、出院日期、手术日期等
- ⚠️ 重要：只标注字段的【值】，绝对不要标注字段的【标签/名称】！
  例如 "姓名：张三" → 只标注 "张三"，不要标注 "姓名："
  例如 "患者姓名：张三" → 只标注 "张三"，category 必须是 "name"
  例如 "受检者：张三" → 只标注 "张三"，category 必须是 "name"
  例如 "联系人：张三" → 只标注 "张三"，category 必须是 "name"
  例如 "采样人：张三" → 只标注 "张三"，category 必须是 "name"
  例如 "报告医生：李四" → 只标注 "李四"，category 必须是 "doctor"
  例如 单据顶部标题 "无锡市第二人民医院南院" → 整行属于医院名称，category 必须是 "hospital"
  例如 "医院名称：无锡市第二人民医院南院" → 只标注 "无锡市第二人民医院南院"，category 必须是 "hospital"
  例如 "检查日期：2026.01.18" → 只标注 "2026.01.18"，category 必须是 "date"
  例如 "报告日期：2026年1月18日" → 只标注 "2026年1月18日"，category 必须是 "date"
  例如 "病理号：C2400286" → 只标注 "C2400286"，不要标注 "病理号："
  如果标签后面没有值（被涂抹/遮挡/空白），则该字段不需要标注任何内容

严格只识别以下几类个人身份隐私字段的【值】（不是标签本身）：
"""

_PROMPT_FOOTER = """
⚠️ OCR 识别可能存在误差（漏字、错字），当图片中可见的姓名与 OCR 文本存在
   1-2 个字的差异时，请返回 OCR 文本中实际存在的最接近子串（而非图片中的完整值）。

【绝对不要标注】：
- 科室名称、报告类型
- 报告标题或栏位标题（如"MR检查报告单"、"报告描述"、"报告诊断"）
- 医生职称（如"主任医师"，只标注姓名部分）
- 诊断结论、病情描述、检查结果
- 检查项目或检查部位（如"MRI眼眶平扫"）
- 性别（男/女）
__DATE_EXCLUSION__
- ⚠️ 标签文字本身（如"姓名："、"病理号："、"住院号："、"送检科室："、"报告医生："等标签绝对不要标注，只标注标签冒号后面的实际值）
- 职称（主任医师、副主任医师等）
- 标签后没有实际内容的字段（如"见习医生："后面没有姓名、"姓名"后面是空白/横线/被涂抹遮挡，则该字段完全不标注）
- 字段值为空格、横线、下划线等占位符的行
- 已被涂抹/遮挡/手写涂黑的字段（图片中已无法辨认的内容不需要再标注）

返回规则：
- line_index: 敏感值所在的 OCR 行 index（必须是 OCR 数据中真实存在的 index）
- text: 需要打码的敏感值文本（必须能在对应行的 text 中找到的子串）
- category: 隐私类别
- 若一行有多个敏感值，分别输出多条

⚠️ 手写签名检测（重要）：
如果图片中存在手写签名（手写体文字，非印刷体），无论 OCR 是否识别到该区域，
都必须额外返回一条记录，格式为：
{"type": "handwritten_signature", "bbox_pct": [x1, y1, x2, y2]}
其中 x1, y1, x2, y2 为签名区域左上角和右下角占图片宽高的百分比（0-100）。
手写签名常见于"手签："、"签名："等标签之后。

- 只返回 JSON 数组，不要任何其他文字，不要 markdown 代码块"""


def build_system_prompt(categories: Optional[List[str]] = None) -> str:
    """根据给定的类别列表，动态拼接 system prompt。"""
    if categories is None:
        categories = DEFAULT_CATEGORIES

    lines = []
    for i, cat in enumerate(categories, 1):
        desc = CATEGORY_DESCRIPTIONS.get(cat, cat)
        lines.append(f"{i}. {cat:15s} - {desc}")

    date_exclusion = (
        ""
        if "date" in categories
        else "- 日期（除出生日期外）"
    )

    footer = _PROMPT_FOOTER.replace("__DATE_EXCLUSION__", date_exclusion)
    return _PROMPT_HEADER + "\n".join(lines) + footer


VISION_USER_PROMPT = """OCR 文本行：
{ocr_json}

请对照图片和上面的 OCR 文本，找出所有敏感信息。
返回格式（只返回 JSON 数组）：
[{{"line_index": OCR行index, "text": "敏感值", "category": "类别"}}]"""


class BaseDetector(ABC):
    """所有 LLM 检测器必须实现此接口。"""

    # 医疗表单中固定字段模式：字段标签 + 冒号 + 姓名值
    _DOCTOR_FIELD_RE = re.compile(
        r'(?:申请|报告|审核|主治|经治|检查|主管|开单|技术|初诊|复诊)?'
        r'(?:医生|医师|技师|操作者|检查者)'
        r'[：:]\s*'
        r'([\u4e00-\u9fff]{2,3}?)'
        r'(?=[：:\s\d床号科室门诊住院检查'
        r'报告审核主治经治申请]|\Z)'
    )

    # 医疗编号类字段模式：标签 + 冒号 + 编号值
    _MEDICAL_ID_RE = re.compile(
        r'(?:报告单号|病历号|就诊号|住院号|超声号|门诊号|检查号|'
        r'登记号|条码号?|检验号|标本号|申请号|流水号|'
        r'检验仪器|检查仪器|仪器号|检验器号|仪器编号)'
        r'[：:]\s*'
        r'([A-Za-z0-9\-_.]{2,})'
    )

    # 采样人/送检人等非医生的人名字段
    _STAFF_NAME_RE = re.compile(
        r'(?:采样人|送检人|送检者|核收人|录入人)'
        r'[：:]\s*'
        r'([\u4e00-\u9fff]{2,4})'
    )

    # 手写签名标签模式：匹配"手签："及其后的所有内容
    _HAND_SIGN_RE = re.compile(r'手签[：:]\s*')

    # 医生标签模式（不要求后面有姓名，用于检测空标签）
    _DOCTOR_LABEL_ONLY_RE = re.compile(
        r'(?:报告录入|报告|审核|主治|经治|检查|申请|主管|开单)?'
        r'(?:医生|医师|技师|录入)'
        r'[：:]\s*$'
    )

    @abstractmethod
    def _call_vision(self, b64_data: str, media_type: str,
                     user_prompt: str, system_prompt: str) -> str:
        """调用 Vision LLM，返回原始文本响应。"""
        ...

    def _extract_rule_based(self, text_lines: List[TextLine]) -> List[SensitiveRegion]:
        """
        基于正则的字段模式匹配，直接从 OCR 文本提取固定结构字段。
        补充 LLM 检测，避免因 OCR 漏字或 LLM 漏判导致遗漏。
        """
        regions = []
        for line in text_lines:
            # 医生姓名
            for m in self._DOCTOR_FIELD_RE.finditer(line.text):
                name = m.group(1)
                polygon = get_value_polygon(line, name, prefix_miss=1)
                regions.append(SensitiveRegion(
                    text=name,
                    category='doctor',
                    polygon=polygon,
                    ocr_line_index=line.index,
                ))
                logger.info(f"规则匹配-医生姓名: '{name}'（含左扩展）在OCR行{line.index}")

            # 医疗编号类（报告单号、条码号、检验号、仪器号等）
            for m in self._MEDICAL_ID_RE.finditer(line.text):
                value = m.group(1)
                polygon = get_value_polygon(line, value)
                regions.append(SensitiveRegion(
                    text=value,
                    category='medical_id',
                    polygon=polygon,
                    ocr_line_index=line.index,
                ))
                logger.info(f"规则匹配-编号: '{value}' 在OCR行{line.index}")

            # 采样人等工作人员姓名
            for m in self._STAFF_NAME_RE.finditer(line.text):
                name = m.group(1)
                polygon = get_value_polygon(line, name)
                regions.append(SensitiveRegion(
                    text=name,
                    category='name',
                    polygon=polygon,
                    ocr_line_index=line.index,
                ))
                logger.info(f"规则匹配-工作人员: '{name}' 在OCR行{line.index}")

        # 手写签名检测
        regions.extend(self._extract_handwritten_signatures(text_lines))

        # 空医生标签后扩展遮罩（手写名字 OCR 读不到的情况）
        regions.extend(self._mask_empty_doctor_labels(text_lines))
        return regions

    def _mask_empty_doctor_labels(
            self, text_lines: List[TextLine]) -> List[SensitiveRegion]:
        """
        当 OCR 行仅包含医生标签（如"检查医生："）而无姓名时，
        说明姓名可能是手写的。向标签右侧扩展遮罩以覆盖手写内容。

        仅当扩展区域内没有高置信度 OCR 文本行（即该区域确实为空/手写）时才生效。
        """
        regions = []
        for line in text_lines:
            if not self._DOCTOR_LABEL_ONLY_RE.search(line.text):
                continue
            # 已经被 _DOCTOR_FIELD_RE 匹配到名字的行跳过
            if self._DOCTOR_FIELD_RE.search(line.text):
                continue

            pts = np.array(line.bbox, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 4:
                continue

            x1, y1, x2, y2 = line.xyxy
            line_h = max(y2 - y1, 1)
            line_w = max(x2 - x1, 1)

            # 扩展区域：标签右端向右延伸 0.7 倍行宽（约 3-4 个汉字宽度）
            ext_x1 = x2
            ext_y1 = y1 - int(line_h * 0.3)
            ext_x2 = x2 + int(line_w * 0.7)
            ext_y2 = y2 + int(line_h * 0.3)

            # 检查扩展区域内是否有高置信度 OCR 文本
            # 如果有，说明该区域是印刷文字（如相邻字段），不应覆盖
            has_printed_text = False
            for other in text_lines:
                if other.index == line.index:
                    continue
                ox1, oy1, ox2, oy2 = other.xyxy
                oy_center = (oy1 + oy2) / 2
                ox_center = (ox1 + ox2) / 2
                if (ext_y1 <= oy_center <= ext_y2 and
                        ext_x1 <= ox_center <= ext_x2 and
                        other.confidence > 0.7 and
                        len(other.text.strip()) >= 2):
                    has_printed_text = True
                    break

            if has_printed_text:
                logger.debug(
                    f"空医生标签扩展跳过（扩展区域有印刷文字）: '{line.text}'"
                )
                continue

            # 构建扩展遮罩多边形
            p_tl, p_tr, p_br, p_bl = pts[0], pts[1], pts[2], pts[3]
            top_vec = p_tr - p_tl
            bottom_vec = p_br - p_bl
            line_w_vec = np.linalg.norm(top_vec)

            extend_len = line_w_vec * 0.7
            dir_top = top_vec / max(line_w_vec, 1)
            dir_bot = bottom_vec / max(np.linalg.norm(bottom_vec), 1)

            new_tl = p_tr.copy()
            new_tr = p_tr + dir_top * extend_len
            new_bl = p_br.copy()
            new_br = p_br + dir_bot * extend_len

            pad_y = line_h * 0.5
            new_tl += np.array([0, -pad_y])
            new_tr += np.array([0, -pad_y])
            new_bl += np.array([0, pad_y])
            new_br += np.array([0, pad_y])

            polygon = np.array([new_tl, new_tr, new_br, new_bl], dtype=np.int32)
            regions.append(SensitiveRegion(
                text='[手写姓名]',
                category='doctor',
                polygon=polygon,
                ocr_line_index=line.index,
            ))
            logger.info(
                f"规则匹配-空医生标签扩展: '{line.text}' OCR行{line.index}"
            )
        return regions

    def _extract_handwritten_signatures(
            self, text_lines: List[TextLine]) -> List[SensitiveRegion]:
        """
        检测"手签："标签后的手写签名区域。

        手写签名通常不被 OCR 识别，需根据"手签："标签位置
        向右大幅扩展遮罩（cv2.fillPoly 自动裁剪到图片边界）。
        """
        regions = []
        for line in text_lines:
            m = self._HAND_SIGN_RE.search(line.text)
            if not m:
                continue

            pts = np.array(line.bbox, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 4:
                continue

            p_tl, p_tr, p_br, p_bl = pts[0], pts[1], pts[2], pts[3]
            top_vec = p_tr - p_tl
            bottom_vec = p_br - p_bl
            line_h = max(p_bl[1] - p_tl[1], p_br[1] - p_tr[1], 1)

            # 签名起点：从"手签："之后开始
            total_len = max(len(line.text), 1)
            start_ratio = m.end() / total_len

            new_tl = p_tl + top_vec * start_ratio
            new_bl = p_bl + bottom_vec * start_ratio

            # 签名终点：向右扩展整行宽度（手写签名可能远超 OCR 行尾）
            # cv2.fillPoly 会自动裁剪超出图片边界的部分
            line_w = np.linalg.norm(top_vec)
            extend_vec_top = top_vec / max(line_w, 1) * line_w
            extend_vec_bot = bottom_vec / max(np.linalg.norm(bottom_vec), 1) * line_w
            new_tr = p_tr + extend_vec_top
            new_br = p_br + extend_vec_bot

            # 上下各扩展半个行高（手写字通常比印刷字更大）
            pad_y = line_h * 0.5
            new_tl = new_tl + np.array([0, -pad_y])
            new_tr = new_tr + np.array([0, -pad_y])
            new_bl = new_bl + np.array([0, pad_y])
            new_br = new_br + np.array([0, pad_y])

            polygon = np.array([new_tl, new_tr, new_br, new_bl], dtype=np.int32)
            regions.append(SensitiveRegion(
                text='[手写签名]',
                category='handwritten_signature',
                polygon=polygon,
                ocr_line_index=line.index,
            ))
            logger.info(f"规则匹配-手写签名: OCR行{line.index}")
        return regions

    def detect(self, image_path: str,
               text_lines: List[TextLine],
               categories: Optional[List[str]] = None) -> List[SensitiveRegion]:
        from ...config import get_settings
        import time

        settings = get_settings()
        prep_started_at = time.time()
        b64_started_at = time.time()
        b64_data, media_type, original_size, sent_size = image_to_base64(
            image_path,
            max_side=settings.VISION_IMAGE_MAX_SIDE,
            jpeg_quality=settings.VISION_JPEG_QUALITY,
        )
        b64_ms = (time.time() - b64_started_at) * 1000

        # 读取图片尺寸（用于手写签名百分比坐标转换）
        from PIL import Image
        image_meta_started_at = time.time()
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        image_meta_ms = (time.time() - image_meta_started_at) * 1000

        prompt_started_at = time.time()
        ocr_data = [{'index': l.index, 'text': l.text} for l in text_lines]
        ocr_json = json.dumps(ocr_data, ensure_ascii=False, indent=2)
        user_prompt = VISION_USER_PROMPT.format(ocr_json=ocr_json)
        system_prompt = build_system_prompt(categories)
        prompt_ms = (time.time() - prompt_started_at) * 1000
        prep_ms = (time.time() - prep_started_at) * 1000

        _perf(
            "Vision 预处理完成: image=%s, original=%sx%s, sent=%sx%s, payload_kb=%s, text_lines=%s, categories=%s, b64_ms=%.1f, image_meta_ms=%.1f, prompt_ms=%.1f, prep_ms=%.1f"
            % (
                image_path,
                original_size[0], original_size[1],
                sent_size[0], sent_size[1],
                len(b64_data) // 1024,
                len(text_lines),
                len(categories) if categories else len(DEFAULT_CATEGORIES),
                b64_ms,
                image_meta_ms,
                prompt_ms,
                prep_ms,
            )
        )

        try:
            call_started_at = time.time()
            raw = self._call_vision(b64_data, media_type, user_prompt, system_prompt)
            call_ms = (time.time() - call_started_at) * 1000
            parse_started_at = time.time()
            llm_regions = self._parse_and_locate(
                raw, text_lines, img_w, img_h, image_path=image_path
            )
            parse_ms = (time.time() - parse_started_at) * 1000
            _perf(
                "Vision LLM 检测到 %s 处敏感信息: call_ms=%.1f, parse_ms=%.1f, response_chars=%s"
                % (len(llm_regions), call_ms, parse_ms, len(raw))
            )
        except Exception as e:
            logger.error(f"Vision LLM 检测失败: {e}")
            raise RuntimeError(f"Vision LLM 检测失败: {e}") from e


        # 合并规则匹配结果
        rule_started_at = time.time()
        rule_regions = self._extract_rule_based(text_lines)
        rule_ms = (time.time() - rule_started_at) * 1000

        # CV 检测底部手写签名（OCR 漏检时的兜底）
        cv_started_at = time.time()
        cv_regions = self._detect_signature_cv(image_path, text_lines)
        cv_ms = (time.time() - cv_started_at) * 1000

        # CV 人脸检测 → portrait regions（不依赖 LLM bbox，始终执行）
        face_started_at = time.time()
        face_regions = self._detect_faces_cv(image_path)
        face_ms = (time.time() - face_started_at) * 1000

        # CV 检测医生名字附近的手写签名
        doctor_regions = [
            r for r in (llm_regions + rule_regions)
            if r.category == 'doctor'
        ]
        adj_started_at = time.time()
        adj_regions = self._detect_signature_near_doctors(
            image_path, text_lines, doctor_regions
        )
        adj_ms = (time.time() - adj_started_at) * 1000

        dedup_started_at = time.time()
        all_regions = self._deduplicate(
            llm_regions + rule_regions + cv_regions + adj_regions + face_regions
        )
        dedup_ms = (time.time() - dedup_started_at) * 1000
        _perf(
            "Vision 后处理完成: llm=%s, rule=%s, cv=%s, adj=%s, face=%s, final=%s, "
            "rule_ms=%.1f, cv_ms=%.1f, adj_ms=%.1f, face_ms=%.1f, dedup_ms=%.1f"
            % (
                len(llm_regions), len(rule_regions), len(cv_regions),
                len(adj_regions), len(face_regions), len(all_regions),
                rule_ms, cv_ms, adj_ms, face_ms, dedup_ms,
            )
        )
        extra = len(rule_regions) + len(cv_regions) + len(adj_regions) + len(face_regions)
        if extra:
            logger.info(f"规则+CV 补充 {extra} 处，合并后共 {len(all_regions)} 处")
        return all_regions

    # ── 医生名字附近手写签名检测 ──

    # 医生姓名标签模式（用于定位签名搜索区域）
    _DOCTOR_LABEL_RE = re.compile(
        r'(?:报告|审核|主治|经治|检查|申请|主管)?'
        r'(?:医生|医师|技师)'
    )

    def _detect_signature_near_doctors(
            self, image_path: str,
            text_lines: List[TextLine],
            doctor_regions: List[SensitiveRegion]) -> List[SensitiveRegion]:
        """
        检测医生名字附近的手写签名。

        医疗单据中，手写签名通常紧跟在印刷体医生姓名的下方或右侧。
        当检测到医生姓名后，扫描其下方区域寻找手写笔迹。
        """
        import cv2

        if not doctor_regions:
            return []

        image = cv2.imread(image_path)
        if image is None:
            return []

        h, w = image.shape[:2]
        regions = []

        # 找到包含医生标签的 OCR 行，确定签名搜索区域
        doctor_lines = set()
        for dr in doctor_regions:
            if dr.ocr_line_index >= 0:
                doctor_lines.add(dr.ocr_line_index)

        # 对每个医生名字，搜索其下方区域
        for dr in doctor_regions:
            x1d, y1d, x2d, y2d = dr.bbox
            line_h = max(y2d - y1d, 20)

            # 搜索区域：医生名字正下方，宽度扩展，高度向下延伸 2 倍行高
            search_x1 = max(0, x1d - int(line_h * 0.5))
            search_y1 = y2d
            search_x2 = min(w, x2d + int(line_h * 0.5))
            search_y2 = min(h, y2d + int(line_h * 2.5))

            # 跳过搜索区域太小的情况
            if search_y2 - search_y1 < 10 or search_x2 - search_x1 < 20:
                continue

            # 检查搜索区域内是否有 OCR 已识别的文本行
            # （如果 OCR 已识别，说明是印刷体，不是手写签名）
            has_ocr_in_area = False
            for line in text_lines:
                lx1, ly1, lx2, ly2 = line.xyxy
                # OCR 行的中心在搜索区域内
                ly_center = (ly1 + ly2) / 2
                lx_center = (lx1 + lx2) / 2
                if (search_y1 <= ly_center <= search_y2 and
                        search_x1 <= lx_center <= search_x2):
                    # 排除低置信度的短文本（可能是 OCR 对手写字的误识别）
                    if line.confidence > 0.7 and len(line.text.strip()) >= 2:
                        has_ocr_in_area = True
                        break

            if has_ocr_in_area:
                continue

            # CV 分析搜索区域的笔迹
            roi = image[search_y1:search_y2, search_x1:search_x2]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            roi_h, roi_w = roi.shape[:2]
            min_area = roi_w * roi_h * 0.005
            stroke_candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                # 排除横线
                if cw / max(ch, 1) > 10:
                    continue
                # 排除竖线
                if ch / max(cw, 1) > 10:
                    continue
                stroke_candidates.append(cnt)

            if len(stroke_candidates) < 2:
                continue

            # 合并笔画候选
            all_pts = np.vstack(stroke_candidates)
            rx, ry, rw, rh = cv2.boundingRect(all_pts)

            # 填充密度检查
            stroke_roi = binary[ry:ry + rh, rx:rx + rw]
            fill_ratio = np.count_nonzero(stroke_roi) / max(rw * rh, 1)
            if fill_ratio > 0.50 or fill_ratio < 0.01:
                continue

            # 尺寸合理性
            if rw < 15 or rh < 8:
                continue

            # 转换回原图坐标
            pad = int(max(rh * 0.3, 5))
            abs_x1 = max(0, search_x1 + rx - pad)
            abs_y1 = max(0, search_y1 + ry - pad)
            abs_x2 = min(w, search_x1 + rx + rw + pad)
            abs_y2 = min(h, search_y1 + ry + rh + pad)

            polygon = np.array(
                [[abs_x1, abs_y1], [abs_x2, abs_y1],
                 [abs_x2, abs_y2], [abs_x1, abs_y2]],
                dtype=np.int32
            )
            regions.append(SensitiveRegion(
                text='[手写签名]',
                category='handwritten_signature',
                polygon=polygon,
                ocr_line_index=-1,
            ))
            logger.info(
                f"CV 检测到医生名字附近手写签名: "
                f"doctor='{dr.text}' sig_bbox=({abs_x1},{abs_y1},{abs_x2},{abs_y2}), "
                f"填充率={fill_ratio:.1%}, 笔画数={len(stroke_candidates)}"
            )

        return regions

    def _detect_faces_cv(self, image_path: str) -> List[SensitiveRegion]:
        """CV 人脸检测 → portrait 类 SensitiveRegion。

        不依赖 LLM 结果，始终对整图运行；空列表表示未检测到人像。
        """
        import cv2
        try:
            from ..face_detector import FaceDetector
        except Exception as e:
            logger.warning(f"人脸检测器加载失败: {e}")
            return []

        image = cv2.imread(image_path)
        if image is None:
            return []

        detector = FaceDetector()
        faces = detector.detect(image)
        if not faces:
            return []

        h, w = image.shape[:2]
        regions: List[SensitiveRegion] = []
        for (x1, y1, x2, y2) in faces:
            # 适度外扩，覆盖头发/帽子/颈部，保证隐私
            pad_w = int((x2 - x1) * 0.20)
            pad_h = int((y2 - y1) * 0.30)
            ex1 = max(0, x1 - pad_w)
            ey1 = max(0, y1 - pad_h)
            ex2 = min(w, x2 + pad_w)
            ey2 = min(h, y2 + pad_h)
            polygon = np.array(
                [[ex1, ey1], [ex2, ey1], [ex2, ey2], [ex1, ey2]],
                dtype=np.int32,
            )
            regions.append(SensitiveRegion(
                text='[人像]',
                category='portrait',
                polygon=polygon,
                ocr_line_index=-1,
            ))
            logger.info(
                f"CV 检测到人像: bbox=({ex1},{ey1},{ex2},{ey2}) "
                f"backend={detector.backend}"
            )
        return regions

    # 签名上下文关键词：OCR 文本中出现时可降低 CV 检测阈值
    _SIGN_CONTEXT_KEYWORDS = ('签名', '手签', '签字', '签章', '盖章')

    def _detect_signature_cv(
            self, image_path: str,
            text_lines: List[TextLine]) -> List[SensitiveRegion]:
        """
        CV 方法检测 OCR 未覆盖区域的手写签名。

        双层策略：
        第一层（宽松）：OCR 文本中存在"签名/手签"关键词 → 中等阈值
        第二层（严格）：无关键词，纯视觉特征 → 高阈值
            - 笔迹簇必须紧凑（宽度 < 图片宽度 40%）
            - 至少 4 个笔画轮廓
            - 填充密度在手写范围内（3%-35%）
            - 不在图片最底边缘（排除边框/按钮栏）
        """
        import cv2

        if not text_lines:
            return []

        image = cv2.imread(image_path)
        if image is None:
            return []

        h, w = image.shape[:2]
        last_y = max(l.xyxy[3] for l in text_lines)
        bottom_margin = h - last_y

        # 底部空间不足 8%，不可能有签名
        if bottom_margin < h * 0.08:
            return []

        # 判断是否有签名上下文
        all_text = ''.join(l.text for l in text_lines)
        has_context = any(kw in all_text for kw in self._SIGN_CONTEXT_KEYWORDS)

        # ── 截取底部区域，内缩左右 3% 避免边缘阴影 ──
        margin_x = int(w * 0.03)
        bottom = image[last_y:h, margin_x:w - margin_x]
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── 逐轮廓过滤（通用） ──
        bh, bw = bottom.shape[:2]
        min_area = h * w * 0.0005
        stroke_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            # 排除太扁的横线（表格线/分隔线）
            if cw / max(ch, 1) > 12:
                continue
            # 排除太窄的竖线（边框）
            if ch / max(cw, 1) > 12:
                continue
            # 排除贴近边缘的横跨型轮廓（屏幕边框/按钮栏）
            if cw > bw * 0.7 and (cy < 3 or cy + ch > bh - 3):
                continue
            stroke_candidates.append(cnt)

        if not stroke_candidates:
            return []

        # 合并所有候选轮廓的外接矩形
        all_pts = np.vstack(stroke_candidates)
        rx, ry, rw, rh = cv2.boundingRect(all_pts)

        # 填充密度
        roi = binary[ry:ry + rh, rx:rx + rw]
        fill_ratio = np.count_nonzero(roi) / max(rw * rh, 1)

        # ── 根据有无上下文采用不同阈值 ──
        if has_context:
            # 第一层：有关键词，适中阈值
            if fill_ratio > 0.45 or fill_ratio < 0.02:
                return []
            if rw * rh > bw * bh * 0.7:
                return []
            if rw < 20 or rh < 10:
                return []
        else:
            # 第二层：无关键词，严格阈值 — 必须像手写签名
            # 1) 笔画数足够（手写签名 ≥ 4 笔画）
            if len(stroke_candidates) < 4:
                return []
            # 2) 笔迹簇紧凑（不横跨全宽，签名通常集中在一个区域）
            if rw > bw * 0.40:
                return []
            # 3) 填充密度在手写范围（太密=印章/色块，太稀=噪点）
            if fill_ratio > 0.35 or fill_ratio < 0.03:
                return []
            # 4) 不在图片最底端（排除屏幕边框/底栏）
            if ry + rh > bh * 0.95:
                return []
            # 5) 尺寸合理
            if rw < 30 or rh < 15:
                return []
            if rw * rh > bw * bh * 0.3:
                return []

        # 转回原图坐标 + padding
        pad = int(max(rh * 0.3, 5))
        x1 = max(0, margin_x + rx - pad)
        y1 = max(last_y, last_y + ry - pad)
        x2 = min(w, margin_x + rx + rw + pad)
        y2 = min(h, last_y + ry + rh + pad)

        polygon = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
        )
        tier = "关键词+CV" if has_context else "纯视觉"
        logger.info(
            f"CV 检测到底部手写签名（{tier}）: bbox=({x1},{y1},{x2},{y2}), "
            f"填充率={fill_ratio:.1%}, 笔画数={len(stroke_candidates)}"
        )
        return [SensitiveRegion(
            text='[手写签名]',
            category='handwritten_signature',
            polygon=polygon,
            ocr_line_index=-1,
        )]

    @staticmethod
    def _verify_handwriting_cv(image_path: str,
                               x1: int, y1: int,
                               x2: int, y2: int) -> bool:
        """
        CV 验证指定区域是否确实包含手写笔迹。
        防止 LLM 返回不准确的 bbox 导致误打码。
        """
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            return False
        h, w = image.shape[:2]
        # 裁剪到图片边界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 5:
            return False

        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        roi_h, roi_w = roi.shape[:2]
        min_area = roi_w * roi_h * 0.003
        strokes = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw / max(ch, 1) > 12 or ch / max(cw, 1) > 12:
                continue
            strokes += 1

        fill = np.count_nonzero(binary) / max(roi_w * roi_h, 1)
        # 手写笔迹特征：至少 2 个笔画，填充率在合理范围
        ok = strokes >= 2 and 0.01 < fill < 0.45
        if not ok:
            logger.info(
                f"手写签名 CV 验证未通过: "
                f"bbox=({x1},{y1},{x2},{y2}) "
                f"strokes={strokes}, fill={fill:.1%}"
            )
        return ok

    def _parse_and_locate(self, raw: str,
                          text_lines: List[TextLine],
                          img_w: int = 0, img_h: int = 0,
                          image_path: str = '') -> List[SensitiveRegion]:
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()

        if not raw or raw == '[]':
            return []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\n原始内容: {raw[:500]}")
            return []

        line_map = {l.index: l for l in text_lines}
        regions = []

        for item in data:
            # ── 手写签名特殊格式 ──
            if item.get('type') == 'handwritten_signature' and 'bbox_pct' in item:
                if img_w > 0 and img_h > 0:
                    try:
                        pct = item['bbox_pct']
                        # LLM 可能返回百分比(0-100) 或像素坐标
                        # 如果任一值 > 100，视为像素坐标直接使用
                        if any(v > 100 for v in pct):
                            logger.info(
                                f"手写签名 bbox_pct 值超过 100（{pct}），"
                                f"按像素坐标处理"
                            )
                            x1, y1, x2, y2 = (
                                int(pct[0]), int(pct[1]),
                                int(pct[2]), int(pct[3]),
                            )
                        else:
                            x1 = int(pct[0] / 100 * img_w)
                            y1 = int(pct[1] / 100 * img_h)
                            x2 = int(pct[2] / 100 * img_w)
                            y2 = int(pct[3] / 100 * img_h)
                        # 确保坐标顺序正确
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        # 适当扩展边界以确保覆盖
                        pad = int(max(x2 - x1, y2 - y1) * 0.15)
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(img_w, x2 + pad)
                        y2 = min(img_h, y2 + pad)
                        # CV 验证：确认区域内确实有手写笔迹
                        if image_path and not self._verify_handwriting_cv(
                                image_path, x1, y1, x2, y2):
                            continue

                        polygon = np.array(
                            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                            dtype=np.int32
                        )
                        regions.append(SensitiveRegion(
                            text='[手写签名]',
                            category='handwritten_signature',
                            polygon=polygon,
                            ocr_line_index=-1,
                        ))
                        logger.info(
                            f"LLM 检测到手写签名: bbox=({x1},{y1},{x2},{y2})"
                        )
                    except (IndexError, TypeError, ValueError) as e:
                        logger.warning(f"手写签名坐标解析失败: {item}, 错误: {e}")
                continue
            try:
                # 兼容 LLM 可能使用 "index" 代替 "line_index"
                line_idx = int(
                    item.get('line_index', item.get('index', -1))
                )
                if line_idx < 0:
                    logger.warning(f"跳过缺少 line_index 的条目: {item}")
                    continue
                text_val = str(item.get('text', '')).strip()
                category = str(item.get('category', 'unknown'))

                if not text_val:
                    continue

                # ── 过滤标签：LLM 可能误将字段标签当作敏感值 ──
                # 标签特征：以中文冒号/英文冒号结尾（如"姓名："、"病理号："）
                if re.match(r'^[\u4e00-\u9fff\w]+[：:]$', text_val):
                    logger.info(f"过滤标签（非敏感值）: '{text_val}'，跳过")
                    continue

                # 如果 LLM 返回 "标签：值" 整体，尝试提取冒号后的值部分
                label_val_match = re.match(
                    r'^[\u4e00-\u9fff\w]+[：:]\s*(.+)$', text_val
                )
                if label_val_match:
                    extracted = label_val_match.group(1).strip()
                    if extracted:
                        logger.info(
                            f"从 '{text_val}' 中提取值部分: '{extracted}'"
                        )
                        text_val = extracted

                # ── 定位 OCR 行 ──
                line = line_map.get(line_idx)
                if line and text_val in line.text:
                    actual_line = line
                    matched_val = text_val
                    prefix_miss = 0
                else:
                    candidates = [l for l in text_lines if text_val in l.text]
                    actual_line = (
                        min(candidates, key=lambda l: abs(l.index - line_idx))
                        if candidates else None
                    )
                    matched_val = text_val
                    prefix_miss = 0

                    if actual_line:
                        logger.info(
                            f"行号修正: '{text_val}' "
                            f"LLM指定行{line_idx} → 实际行{actual_line.index}"
                        )
                    else:
                        # ── 前缀容错匹配：OCR 可能漏识首字 ──
                        for skip in range(1, min(3, len(text_val) - 1)):
                            partial = text_val[skip:]
                            if len(partial) < 2:
                                break
                            cands = [l for l in text_lines if partial in l.text]
                            if cands:
                                actual_line = min(
                                    cands,
                                    key=lambda l: abs(l.index - line_idx)
                                )
                                matched_val = partial
                                prefix_miss = skip
                                logger.info(
                                    f"前缀容错匹配: '{text_val}' → '{partial}'"
                                    f"（漏识前{skip}字），OCR行{actual_line.index}"
                                )
                                break

                    if actual_line is None:
                        logger.warning(f"OCR 中找不到 '{text_val}'，跳过")
                        continue

                polygon = get_value_polygon(
                    actual_line, matched_val, prefix_miss=prefix_miss
                )

                regions.append(SensitiveRegion(
                    text=text_val,
                    category=category,
                    polygon=polygon,
                    ocr_line_index=actual_line.index,
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"跳过无效条目: {item}，错误: {e}")

        return self._deduplicate(regions)

    @staticmethod
    def _deduplicate(regions: List[SensitiveRegion]) -> List[SensitiveRegion]:
        def _area(r):
            b = r.bbox
            return (b[2] - b[0]) * (b[3] - b[1])

        regions.sort(key=lambda r: -_area(r))
        result = []
        for r in regions:
            overlap = False
            rb = r.bbox
            for u in result:
                ub = u.bbox
                ix1 = max(rb[0], ub[0])
                iy1 = max(rb[1], ub[1])
                ix2 = min(rb[2], ub[2])
                iy2 = min(rb[3], ub[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = _area(r) + _area(u) - inter
                    if union > 0 and inter / union > 0.3:
                        overlap = True
                        break
            if not overlap:
                result.append(r)
        return result
