"""
本地图片打码测试
================
对 tests/input/ 下的图片分别用 volcengine 和 volcengine-lite 打码，
结果输出至 tests/output/<provider>/<原文件名>_masked<后缀>，方便对比精度。

运行方式：
  pytest tests/test_masking.py -v
  python tests/test_masking.py
"""

import logging
import os
import queue
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

logger = logging.getLogger(__name__)

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"
_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
_WORKERS = 4


def _collect_images():
    if not INPUT_DIR.exists():
        return []
    return sorted(f for f in INPUT_DIR.iterdir()
                  if f.is_file() and f.suffix.lower() in _EXTS)


def _build_pipeline(provider: str):
    from app.core.ocr.paddle import PaddleOCREngine
    from app.core.masking.solid import SolidColorStrategy
    from app.core.pipeline import MedicalPrivacyMasker
    from app.core.detectors.volcengine import VolcEngineDetector, VolcEngineDetectorLite

    if provider == "volcengine-lite":
        detector = VolcEngineDetectorLite()
    else:
        detector = VolcEngineDetector()

    return MedicalPrivacyMasker(
        ocr=PaddleOCREngine(),
        detector=detector,
        mask_strategy=SolidColorStrategy(color=(0, 0, 255)),  # BGR: 红色
    )


def _run_provider(provider: str, images: list) -> dict:
    """用指定 provider 处理所有图片，返回 {image.name: (out_path, err)}。"""
    out_dir = OUTPUT_DIR / provider
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {img.name: out_dir / f"{img.stem}_masked{img.suffix}" for img in images}
    results = {}

    n = min(_WORKERS, len(images))
    pool: queue.Queue = queue.Queue()
    for i in range(n):
        logger.info(f"[{provider}] 初始化 pipeline {i + 1}/{n}…")
        pool.put(_build_pipeline(provider))

    def _process(img: Path):
        pipeline = pool.get()
        try:
            pipeline.process_image(str(img), str(out_paths[img.name]))
            results[img.name] = (out_paths[img.name], None)
            logger.info(f"[{provider}] 完成: {img.name}")
        except Exception as e:
            results[img.name] = (out_paths[img.name], e)
            logger.error(f"[{provider}] 失败: {img.name} — {e}")
        finally:
            pool.put(pipeline)

    with ThreadPoolExecutor(max_workers=n) as executor:
        for f in as_completed([executor.submit(_process, img) for img in images]):
            f.result()

    return results


class TestVolcengine(unittest.TestCase):
    _results: dict = {}
    _images: list = []

    @classmethod
    def setUpClass(cls):
        cls._images = _collect_images()
        if not cls._images:
            raise unittest.SkipTest("tests/input/ 中没有图片")
        if not os.getenv("ARK_API_KEY"):
            raise unittest.SkipTest("未配置 ARK_API_KEY")
        cls._results = _run_provider("volcengine", cls._images)

    def _assert(self, img: Path):
        out_path, err = self._results.get(img.name, (None, None))
        if err:
            self.fail(f"异常: {err}")
        self.assertTrue(out_path and out_path.exists(), f"输出文件不存在: {out_path}")
        src = cv2.imread(str(img))
        dst = cv2.imread(str(out_path))
        self.assertIsNotNone(dst, "输出图片无法读取")
        self.assertEqual(src.shape[0] * src.shape[1], dst.shape[0] * dst.shape[1],
                         f"像素数不一致: 输入{src.shape[:2]} 输出{dst.shape[:2]}")


class TestVolcengineLite(unittest.TestCase):
    _results: dict = {}
    _images: list = []

    @classmethod
    def setUpClass(cls):
        cls._images = _collect_images()
        if not cls._images:
            raise unittest.SkipTest("tests/input/ 中没有图片")
        if not os.getenv("ARK_API_KEY"):
            raise unittest.SkipTest("未配置 ARK_API_KEY")
        cls._results = _run_provider("volcengine-lite", cls._images)

    def _assert(self, img: Path):
        out_path, err = self._results.get(img.name, (None, None))
        if err:
            self.fail(f"异常: {err}")
        self.assertTrue(out_path and out_path.exists(), f"输出文件不存在: {out_path}")
        src = cv2.imread(str(img))
        dst = cv2.imread(str(out_path))
        self.assertIsNotNone(dst, "输出图片无法读取")
        self.assertEqual(src.shape[0] * src.shape[1], dst.shape[0] * dst.shape[1],
                         f"像素数不一致: 输入{src.shape[:2]} 输出{dst.shape[:2]}")


def _make_test(img: Path):
    def _test(self):
        self._assert(img)
    _test.__name__ = f"test_{img.stem}"
    _test.__doc__ = img.name
    return _test


for _img in _collect_images():
    setattr(TestVolcengine, f"test_{_img.stem}", _make_test(_img))
    setattr(TestVolcengineLite, f"test_{_img.stem}", _make_test(_img))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    unittest.main(verbosity=2)