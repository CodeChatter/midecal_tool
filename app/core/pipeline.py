"""编排器 — MedicalPrivacyMasker（依赖注入）"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .models import ProcessStats, SensitiveRegion, TextLine
from .ocr.base import BaseOCREngine
from .detectors.base import BaseDetector
from .masking.base import BaseMaskStrategy

logger = logging.getLogger(__name__)


def _perf(msg: str):
    logger.info(msg)
    print(f"[PERF] {msg}", file=sys.stderr, flush=True)


# ============================================================
# 图片方向矫正
# ============================================================

class OrientationCorrector:
    """
    图片方向自动矫正。

    策略（按优先级）：
    1. EXIF Orientation 标签（手机拍照，零成本）
    2. 水平投影方差分析（无 EXIF 的扫描件/截图）
       原理：横排文字产生"行密 → 行稀"的投影峰谷，方差最大的旋转即为正方向。

    返回的角度为"顺时针旋转度数"（0 / 90 / 180 / 270）。
    """

    _EXIF_TO_ANGLE: dict = {3: 180, 6: 90, 8: 270}

    _CV2_ROTATE: dict = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    # 投影方差分析用的缩略图最长边，小图跑得快，精度足够
    _ORIENT_THUMB_SIDE = 800

    def correct(self, image_path: str) -> Tuple[np.ndarray, int]:
        """
        检测图片方向并返回矫正后的图像。
        图片只读取一次，同一份内存同时用于方向检测和旋转矫正。

        Returns:
            (corrected_bgr_image, angle_applied)
            angle_applied: 顺时针旋转度数，0 表示无需旋转
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        angle = self._detect_angle(image_path, image)

        if angle in self._CV2_ROTATE:
            image = cv2.rotate(image, self._CV2_ROTATE[angle])
            logger.info(f"方向矫正：图片顺时针旋转 {angle}°")
        else:
            logger.info("方向检测：图片方向正常，无需旋转")

        return image, angle

    def _detect_angle(self, image_path: str, image: np.ndarray = None) -> int:
        angle = self._exif_angle(image_path)
        if angle != 0:
            logger.info(f"方向来源：EXIF（旋转 {angle}°）")
            return angle

        angle = self._projection_angle(image)
        if angle != 0:
            logger.info(f"方向来源：投影方差分析（旋转 {angle}°）")
        return angle

    def _exif_angle(self, image_path: str) -> int:
        """读取 EXIF Orientation 标签，返回矫正所需的顺时针旋转角度。"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                exif = img._getexif()
                if exif:
                    orientation = exif.get(274)
                    return self._EXIF_TO_ANGLE.get(orientation, 0)
        except Exception as e:
            logger.debug(f"EXIF 读取失败（跳过）: {e}")
        return 0

    def _projection_angle(self, image: np.ndarray = None) -> int:
        """
        水平投影方差分析：仅检测 0°/180° 两种方向（上下颠倒）。

        ⚠️ 为何不检测 90°/270°：
          投影方差对宽幅横向图（w > h）会系统性误判——横向排列的多列文字
          旋转 90° 后行方差反而更大，导致把正向宽幅图错误翻转。
          90°/270° 旋转（手机/扫描仪方向错误）通常有 EXIF 信息，
          应由 _exif_angle() 处理，投影方差无需介入。

        使用缩略图（最长边 _ORIENT_THUMB_SIDE）加速计算。
        """
        try:
            if image is None:
                return 0

            # 缩到小图再分析（快 N 倍，结果不变）
            h, w = image.shape[:2]
            max_side = max(h, w)
            if max_side > self._ORIENT_THUMB_SIDE:
                s = self._ORIENT_THUMB_SIDE / max_side
                thumb = cv2.resize(image,
                                   (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA)
            else:
                thumb = image

            gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
            # 反色二值化：文字像素 = 255，背景 = 0
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # 只比较 0° 和 180°（上下颠倒），不涉及 90°/270°
            var_0 = float(binary.sum(axis=1).var())
            var_180 = float(cv2.rotate(binary, cv2.ROTATE_180).sum(axis=1).var())
            logger.debug(f"投影方差 0°={var_0:.0f}, 180°={var_180:.0f}")

            # 180° 明显优于 0° 才矫正（阈值 20%）
            if var_180 > var_0 * 1.20:
                return 180
            return 0

        except Exception as e:
            logger.debug(f"投影方差分析失败（跳过）: {e}")
        return 0


class MedicalPrivacyMasker:
    """
    OCR + Vision LLM 混合脱敏处理器。

    通过依赖注入接收 ocr、detector、mask_strategy 实例，
    不再自己创建具体实现。

    新增功能：
    - 自动矫正图片方向（EXIF + 投影方差分析）
    - OCR 质量验证，乱码时翻转 180° 重试
    """

    def __init__(
            self,
            ocr: BaseOCREngine,
            detector: BaseDetector,
            mask_strategy: BaseMaskStrategy,
            auto_orient: bool = True,
    ):
        self.ocr = ocr
        self.detector = detector
        self.mask_strategy = mask_strategy
        self.auto_orient = auto_orient
        if auto_orient:
            self.orientation_corrector = OrientationCorrector()

    @staticmethod
    def _ocr_looks_valid(text_lines: List[TextLine]) -> bool:
        """
        判断 OCR 结果是否正常（非旋转乱码）。
        乱码特征：行数极少、每行平均字符数 < 3（单字/半字碎片）。
        """
        if len(text_lines) < 4:
            return False
        avg_len = sum(len(l.text) for l in text_lines) / len(text_lines)
        long_lines = sum(1 for l in text_lines if len(l.text) >= 4)
        return avg_len >= 3.0 and long_lines >= 2

    @staticmethod
    def _text_looks_vertical(text_lines: List[TextLine]) -> bool:
        """
        判断 OCR 文本行是否以竖向为主（图片可能旋转了 90°/270°）。

        横向文本行 bbox：宽度 >> 高度
        竖向文本行 bbox：高度 >> 宽度（图片旋转后文字沿纵轴排列）

        当超过半数多字文本行的高度显著大于宽度时，判定为竖向文本。
        只考虑 ≥3 字符的行，过滤掉短文本（标点、页码等本身可能接近正方形）。
        """
        long_lines = [l for l in text_lines if len(l.text.strip()) >= 3]
        if len(long_lines) < 3:
            return False

        vertical_count = 0
        for line in long_lines:
            x1, y1, x2, y2 = line.xyxy
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            if h > w * 1.5:
                vertical_count += 1

        ratio = vertical_count / len(long_lines)
        return ratio > 0.5

    @staticmethod
    def _has_meaningful_text(text_lines: List[TextLine]) -> bool:
        """
        判断图片是否包含足够的有意义文字，值得进行脱敏分析。

        对于无文字图片（CT/MRI/超声影像图、自然场景照片等），
        OCR 可能识别出零星噪声行，但内容极少。
        判定标准：至少 2 行文本，且至少有 1 行包含 3 个以上字符。
        """
        if len(text_lines) < 2:
            return False
        return any(len(l.text.strip()) >= 3 for l in text_lines)

    @staticmethod
    def _copy_to_output(work_path: str, preloaded_image,
                        original_path: str, output_path: Optional[str]):
        """
        无需脱敏时，将图片原样输出到 output_path。
        适用于医学影像图（CT/MRI/超声）等无文字单据。
        """
        if output_path is None:
            p = Path(original_path)
            output_path = str(p.parent / f"{p.stem}_masked_vision{p.suffix}")

        if preloaded_image is not None:
            cv2.imwrite(output_path, preloaded_image)
        else:
            import shutil
            shutil.copy2(work_path, output_path)
        logger.info(f"影像图直接输出（未做脱敏）: {output_path}")

    def process_image(self, image_path: str,
                      output_path: str = None,
                      categories: Optional[List[str]] = None) -> ProcessStats:
        stats = ProcessStats()
        logger.info(f"\n处理图片: {image_path}")

        # ── 阶段 0：方向矫正 ──
        work_path = image_path
        corrected_image = None
        tmp_path = None

        if self.auto_orient:
            orientation_started_at = time.time()
            corrected_image, rotation = self.orientation_corrector.correct(image_path)
            stats.orientation_ms = (time.time() - orientation_started_at) * 1000
            logger.info("[Pipeline] 方向矫正完成: rotation=%s, orientation_ms=%.1f", rotation, stats.orientation_ms)
            if rotation != 0:
                suffix = Path(image_path).suffix or '.jpg'
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                os.close(tmp_fd)
                cv2.imwrite(tmp_path, corrected_image)
                work_path = tmp_path
                logger.info(f"使用方向矫正后的临时图片: {tmp_path}")

        try:
            return self._process_work_image(
                work_path, corrected_image, image_path, output_path, stats, categories
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.debug(f"已清理临时文件: {tmp_path}")

    def _process_work_image(
            self,
            work_path: str,
            preloaded_image: Optional[np.ndarray],
            original_path: str,
            output_path: Optional[str],
            stats: ProcessStats,
            categories: Optional[List[str]],
    ) -> ProcessStats:
        pipeline_started_at = time.time()

        # ── 阶段 1：OCR 提取文本行 + 精确 bbox ──
        t0 = time.time()
        logger.info("[Pipeline] OCR 开始")
        text_lines = self.ocr.recognize(work_path)
        stats.ocr_ms += (time.time() - t0) * 1000
        _perf(f"[Pipeline] OCR 完成: total_lines={len(text_lines)}, ocr_ms={stats.ocr_ms:.1f}")
        stats.total_lines = len(text_lines)

        if not text_lines:
            logger.warning("OCR 未识别到任何文本（可能为医学影像图，非文字单据），跳过脱敏，原图直接输出")
            self._copy_to_output(work_path, preloaded_image, original_path, output_path)
            stats.pipeline_ms = (time.time() - pipeline_started_at) * 1000
            return stats

        # ── 阶段 1.2：早期无效图片判断（快速退出，避免无意义的翻转重试） ──
        if not self._has_meaningful_text(text_lines):
            logger.warning(
                f"初始 OCR 有效文本不足（{len(text_lines)} 行），"
                "判断为非文字单据，跳过脱敏，原图直接输出"
            )
            self._copy_to_output(work_path, preloaded_image, original_path, output_path)
            stats.pipeline_ms = (time.time() - pipeline_started_at) * 1000
            return stats

        # ── 阶段 1.5：OCR 方向验证 + 旋转矫正 ──
        is_vertical = self._text_looks_vertical(text_lines)
        ocr_valid = self._ocr_looks_valid(text_lines)
        avg_len = sum(len(l.text) for l in text_lines) / max(len(text_lines), 1)
        ocr_confident = (
            ocr_valid and not is_vertical and len(text_lines) >= 10
        )
        need_rotation = not ocr_confident and (not ocr_valid or is_vertical)
        _perf(
            f"[Pipeline] OCR 质量评估: total_lines={len(text_lines)}, avg_len={avg_len:.1f}, "
            f"ocr_valid={ocr_valid}, is_vertical={is_vertical}, need_rotation={need_rotation}"
        )

        tmp_flip_path = None
        if self.auto_orient and need_rotation:
            rotation_started_at = time.time()
            reason = "竖向文本排列" if is_vertical else "OCR 结果疑似乱码"
            logger.info(
                f"检测到{reason}（{len(text_lines)} 行，"
                f"均长 {avg_len:.1f} 字），"
                f"尝试旋转矫正…"
            )
            base_img = preloaded_image if preloaded_image is not None else cv2.imread(work_path)

            _ROTATIONS = [
                (cv2.ROTATE_90_CLOCKWISE, "90"),
                (cv2.ROTATE_90_COUNTERCLOCKWISE, "270"),
                (cv2.ROTATE_180, "180"),
            ]

            best_lines = text_lines
            best_image = None
            best_tmp_path = None
            best_vertical = is_vertical

            for cv2_code, angle_label in _ROTATIONS:
                rotated = cv2.rotate(base_img, cv2_code)
                suffix = Path(work_path).suffix or '.jpg'
                tmp_fd, tmp_rot_path = tempfile.mkstemp(suffix=suffix)
                os.close(tmp_fd)
                retry_started_at = time.time()
                try:
                    cv2.imwrite(tmp_rot_path, rotated)
                    retry_lines = self.ocr.recognize(tmp_rot_path)
                    retry_ms = (time.time() - retry_started_at) * 1000
                    retry_vertical = self._text_looks_vertical(retry_lines)
                    retry_valid = self._ocr_looks_valid(retry_lines)

                    improved = False
                    if best_vertical and not retry_vertical and len(retry_lines) >= 3:
                        improved = True
                    elif (
                        not best_vertical and not retry_vertical
                        and len(retry_lines) > len(best_lines)
                    ):
                        improved = True
                    elif (
                        retry_valid
                        and not self._ocr_looks_valid(best_lines)
                    ):
                        improved = True
                    elif len(retry_lines) > len(best_lines) * 1.5:
                        improved = True

                    _perf(
                        f"[Pipeline] OCR 旋转重试: angle={angle_label}, retry_ms={retry_ms:.1f}, "
                        f"lines={len(retry_lines)}, vertical={retry_vertical}, valid={retry_valid}, improved={improved}"
                    )

                    if improved:
                        if best_tmp_path and os.path.exists(best_tmp_path):
                            os.unlink(best_tmp_path)
                        best_lines = retry_lines
                        best_image = rotated
                        best_tmp_path = tmp_rot_path
                        best_vertical = retry_vertical
                        logger.info(
                            f"旋转 {angle_label}° 后 OCR 改善："
                            f"{len(text_lines)} → {len(retry_lines)} 行"
                            f"{'，文本方向已矫正为横向' if not retry_vertical else ''}"
                        )
                        if not retry_vertical and retry_valid:
                            if not is_vertical or angle_label == "270":
                                break
                    else:
                        os.unlink(tmp_rot_path)
                except Exception as e:
                    logger.warning(f"旋转 {angle_label}° 重试失败: {e}")
                    if os.path.exists(tmp_rot_path):
                        os.unlink(tmp_rot_path)

            stats.rotation_retry_ms = (time.time() - rotation_started_at) * 1000
            _perf(f"[Pipeline] OCR 旋转重试完成: rotation_retry_ms={stats.rotation_retry_ms:.1f}")

            if best_image is not None:
                text_lines = best_lines
                stats.total_lines = len(text_lines)
                preloaded_image = best_image
                work_path = best_tmp_path
                tmp_flip_path = best_tmp_path
            else:
                logger.info("所有旋转均无改善，保持原方向")

        # ── 阶段 1.6：最终有效性验证 ──
        if not self._has_meaningful_text(text_lines):
            logger.warning(
                f"图片中有效文本不足（OCR 共 {len(text_lines)} 行，"
                f"均长 {sum(len(l.text) for l in text_lines) / max(len(text_lines), 1):.1f} 字），"
                "判断为非文字单据，跳过脱敏，原图直接输出"
            )
            try:
                self._copy_to_output(work_path, preloaded_image, original_path, output_path)
            finally:
                if tmp_flip_path and os.path.exists(tmp_flip_path):
                    os.unlink(tmp_flip_path)
            stats.pipeline_ms = (time.time() - pipeline_started_at) * 1000
            return stats

        try:
            result = self._do_detect_and_mask(
                work_path, preloaded_image, original_path, output_path, stats, text_lines, categories
            )
            result.pipeline_ms = (time.time() - pipeline_started_at) * 1000
            logger.info(
                "[Pipeline] 完成: total_lines=%s, sensitive=%s, ocr_ms=%.1f, rotation_retry_ms=%.1f, detect_ms=%.1f, mask_ms=%.1f, pipeline_ms=%.1f",
                result.total_lines,
                result.sensitive_count,
                result.ocr_ms,
                result.rotation_retry_ms,
                result.detect_ms,
                result.mask_ms,
                result.pipeline_ms,
            )
            return result
        finally:
            if tmp_flip_path and os.path.exists(tmp_flip_path):
                os.unlink(tmp_flip_path)

    def _do_detect_and_mask(
            self,
            work_path: str,
            preloaded_image: Optional[np.ndarray],
            original_path: str,
            output_path: Optional[str],
            stats: ProcessStats,
            text_lines: List[TextLine],
            categories: Optional[List[str]],
    ) -> ProcessStats:
        for line in text_lines:
            logger.info(f"  OCR 行{line.index:02d}: {line.text}")

        # ── 阶段 2：Vision LLM 看图 + OCR 文本 → 敏感区域 ──
        t0 = time.time()
        _perf(f"[Pipeline] Vision LLM 检测开始: categories={categories if categories else 'default'}")
        regions = self.detector.detect(work_path, text_lines, categories)
        stats.detect_ms = (time.time() - t0) * 1000
        _perf(f"[Pipeline] Vision LLM 检测完成: sensitive={len(regions)}, detect_ms={stats.detect_ms:.1f}")
        stats.sensitive_count = len(regions)

        if not regions:
            logger.info("未检测到敏感信息，无需脱敏")
            copy_started_at = time.time()
            self._copy_to_output(work_path, preloaded_image, original_path, output_path)
            _perf(f"[Pipeline] 原图直出完成: copy_ms={(time.time() - copy_started_at) * 1000:.1f}")
            return stats

        stats.sensitive_items = [r.text for r in regions]
        logger.info(f"检测到敏感信息: {stats.sensitive_items}")

        # ── 阶段 3：遮罩 ──
        t0 = time.time()
        image = preloaded_image if preloaded_image is not None else cv2.imread(work_path)
        if image is None:
            raise ValueError(f"无法读取图片: {work_path}")

        for region in regions:
            self.mask_strategy.apply(image, region.polygon)
            stats.masked_count += 1
            x1, y1, x2, y2 = region.bbox
            logger.info(
                f"遮罩: '{region.text}' ({region.category}) "
                f"OCR行{region.ocr_line_index} "
                f"bbox=({x1},{y1},{x2},{y2})"
            )

        if output_path is None:
            p = Path(original_path)
            output_path = str(p.parent / f"{p.stem}_masked_vision{p.suffix}")

        write_started_at = time.time()
        cv2.imwrite(output_path, image)
        write_ms = (time.time() - write_started_at) * 1000
        stats.mask_ms = (time.time() - t0) * 1000
        _perf(f"[Pipeline] 遮罩完成: masked={stats.masked_count}, mask_ms={stats.mask_ms:.1f}, write_ms={write_ms:.1f}, output={output_path}")
        return stats

    def detect_sensitive(self, image_path: str,
                         categories: Optional[List[str]] = None) -> Tuple[List[SensitiveRegion], ProcessStats]:
        """仅检测敏感信息，不打码。返回 (敏感区域列表, 统计信息)。"""
        stats = ProcessStats()
        logger.info(f"\n检测敏感信息: {image_path}")

        # ── 方向矫正 ──
        work_path = image_path
        tmp_path = None

        if self.auto_orient:
            orientation_started_at = time.time()
            corrected_image, rotation = self.orientation_corrector.correct(image_path)
            stats.orientation_ms = (time.time() - orientation_started_at) * 1000
            logger.info("[Detect] 方向矫正完成: rotation=%s, orientation_ms=%.1f", rotation, stats.orientation_ms)
            if rotation != 0:
                suffix = Path(image_path).suffix or '.jpg'
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                os.close(tmp_fd)
                cv2.imwrite(tmp_path, corrected_image)
                work_path = tmp_path

        try:
            t0 = time.time()
            text_lines = self.ocr.recognize(work_path)
            stats.ocr_ms = (time.time() - t0) * 1000
            stats.total_lines = len(text_lines)
            logger.info("[Detect] OCR 完成: total_lines=%s, ocr_ms=%.1f", stats.total_lines, stats.ocr_ms)

            if not text_lines:
                logger.warning("OCR 未识别到任何文本")
                stats.pipeline_ms = stats.orientation_ms + stats.ocr_ms
                return [], stats

            t0 = time.time()
            regions = self.detector.detect(work_path, text_lines, categories)
            stats.detect_ms = (time.time() - t0) * 1000
            stats.sensitive_count = len(regions)
            stats.sensitive_items = [r.text for r in regions]
            stats.pipeline_ms = stats.orientation_ms + stats.ocr_ms + stats.detect_ms
            logger.info(
                "[Detect] 完成: total_lines=%s, sensitive=%s, orientation_ms=%.1f, ocr_ms=%.1f, detect_ms=%.1f, pipeline_ms=%.1f",
                stats.total_lines,
                stats.sensitive_count,
                stats.orientation_ms,
                stats.ocr_ms,
                stats.detect_ms,
                stats.pipeline_ms,
            )

            return regions, stats
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
