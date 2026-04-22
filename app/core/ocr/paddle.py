"""PaddleOCR 引擎实现 — 子进程隔离模式

OCR 推理在独立子进程中执行，即使 PaddlePaddle C++ 层触发 SIGSEGV，
也只会杀死子进程，主进程（uvicorn worker）存活并返回可控错误。
"""

import logging
import multiprocessing
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ...bootstrap.runtime import bootstrap_runtime
from ..errors import SystemOverloadError
from ..models import TextLine
from ..registry import ocr_registry
from .base import BaseOCREngine

logger = logging.getLogger(__name__)


def _perf(msg: str):
    logger.info(msg)
    print(f"[PERF] {msg}", file=sys.stderr, flush=True)


_OVERLOAD_MARKERS = (
    'out of memory',
    'resourceexhausted',
    'cannot allocate',
    'cublas_status',
    'cudnn_status',
)


def _is_overload_error(message: str) -> bool:
    m = (message or '').lower()
    return any(marker in m for marker in _OVERLOAD_MARKERS)


# ── 子进程 Worker 函数（模块级，可被 pickle）──

_MODEL_VARIANTS = {
    'mobile': ('PP-OCRv5_mobile_det', 'PP-OCRv5_mobile_rec'),
    'server': ('PP-OCRv5_server_det', 'PP-OCRv5_server_rec'),
    'legacy': (None, None),
}


def _worker_init_ocr(variant: str):
    """在子进程中初始化 PaddleOCR 实例。"""
    bootstrap_runtime()
    import paddle
    from paddleocr import PaddleOCR

    install_gpu = os.environ.get('INSTALL_GPU', '')
    paddle_cuda = paddle.device.is_compiled_with_cuda()
    paddle_device = paddle.device.get_device()
    logger.info(
        "OCR worker 初始化: pid=%s, variant=%s, INSTALL_GPU=%s, paddle_cuda=%s, paddle_device=%s",
        os.getpid(),
        variant,
        install_gpu,
        paddle_cuda,
        paddle_device,
    )
    if install_gpu.lower() == 'true' and (not paddle_cuda or paddle_device.startswith('cpu')):
        logger.warning(
            "GPU 模式已启用，但 OCR worker 未运行在 GPU 上: pid=%s, paddle_cuda=%s, paddle_device=%s",
            os.getpid(),
            paddle_cuda,
            paddle_device,
        )

    det_model, rec_model = _MODEL_VARIANTS.get(variant, _MODEL_VARIANTS['server'])
    common_kwargs = dict(lang='ch', enable_mkldnn=False, enable_hpi=False)

    if det_model is None:
        try:
            ocr = PaddleOCR(use_angle_cls=True, **common_kwargs)
        except Exception:
            ocr = PaddleOCR(**common_kwargs)
    else:
        try:
            ocr = PaddleOCR(
                text_detection_model_name=det_model,
                text_recognition_model_name=rec_model,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                **common_kwargs,
            )
        except TypeError:
            try:
                ocr = PaddleOCR(use_angle_cls=True, **common_kwargs)
            except Exception:
                ocr = PaddleOCR(**common_kwargs)

    logger.info(
        "OCR backend ready: pid=%s, variant=%s, paddle_device=%s",
        os.getpid(),
        variant,
        paddle_device,
    )
    return ocr


def _worker_run_ocr(ocr_instance, image_path: str):
    """在子进程中执行 OCR，返回纯 Python 类型结果（可 pickle）。

    关键：优先使用 predict() API（PaddleX 管线），遇到问题会抛 Python 异常；
    ocr() API 在部分 Docker 环境 + v4 模型下会触发 C++ SIGSEGV，放在最后兜底。

    API 兼容性异常（TypeError/AttributeError）触发兜底；
    运行时异常（如显存 OOM）在所有路径都失败时抛出，由上层判定为 OCR 失败。
    """
    result = None
    last_error: Optional[Exception] = None

    # 1) predict() — PaddleX 管线，异常可捕获
    try:
        result = ocr_instance.predict(image_path)
    except (TypeError, AttributeError):
        pass
    except Exception as e:
        last_error = e
        logging.getLogger(__name__).warning(f"predict() 失败: {e}")

    # 2) ocr(cls=True) — 旧版 API 兜底
    if result is None or (isinstance(result, list) and not result):
        try:
            result = ocr_instance.ocr(image_path, cls=True)
        except TypeError:
            try:
                result = ocr_instance.ocr(image_path)
            except Exception as e:
                last_error = e
                result = None
        except Exception as e:
            last_error = e
            result = None

    if result is None:
        if last_error is not None:
            raise RuntimeError(f"OCR 推理失败: {last_error}") from last_error
        return None

    if not result or not result[0]:
        return None

    # 转为纯 Python 类型（确保可跨进程 pickle）
    first = result[0]
    serializable = []

    # 新版 API（dict-like）
    if hasattr(first, '__getitem__') and not isinstance(first, list):
        try:
            texts = first['rec_texts']
            polys = first['dt_polys']
            scores = first.get('rec_scores', [])
            for i, text in enumerate(texts):
                if text and text.strip() and i < len(polys):
                    poly = polys[i].tolist() if hasattr(polys[i], 'tolist') else list(polys[i])
                    score = float(scores[i]) if i < len(scores) else 0.0
                    serializable.append(('new', text, poly, score))
            if serializable:
                return serializable
        except (KeyError, TypeError):
            pass

    # 旧版 API（list of list）
    if isinstance(first, list):
        for line in first:
            if len(line) >= 2:
                poly = line[0].tolist() if hasattr(line[0], 'tolist') else list(line[0])
                ti = line[1]
                text = ti[0] if isinstance(ti, (list, tuple)) else str(ti)
                conf = float(ti[1]) if isinstance(ti, (list, tuple)) and len(ti) > 1 else 0.0
                if text.strip():
                    serializable.append(('old', text, poly, conf))

    return serializable if serializable else None


def _worker_loop(req_queue, resp_queue, variant: str):
    """子进程主循环：初始化模型后持续处理 OCR 请求。"""
    try:
        ocr_instance = _worker_init_ocr(variant)
    except Exception as e:
        resp_queue.put(('init_error', str(e)))
        return

    import paddle
    gpu_active = False
    try:
        gpu_active = (paddle.device.is_compiled_with_cuda()
                      and not paddle.device.get_device().startswith('cpu'))
    except Exception:
        gpu_active = False

    def _release_gpu_cache():
        if not gpu_active:
            return
        try:
            paddle.device.cuda.empty_cache()
        except Exception as exc:
            logging.getLogger(__name__).debug(f"empty_cache 失败: {exc}")

    resp_queue.put(('init_ok', None))

    while True:
        try:
            msg = req_queue.get(timeout=300)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            break

        if msg is None:  # 退出信号
            break

        cmd, payload = msg
        if cmd == 'ocr':
            try:
                result = _worker_run_ocr(ocr_instance, payload)
                resp_queue.put(('ok', result))
            except Exception as e:
                resp_queue.put(('error', str(e)))
            finally:
                _release_gpu_cache()
        elif cmd == 'warmup':
            try:
                _worker_run_ocr(ocr_instance, payload)
                resp_queue.put(('ok', None))
            except Exception as e:
                resp_queue.put(('warmup_error', str(e)))
            finally:
                _release_gpu_cache()


# ── 子进程管理器 ─────────────────────────────────────────

class _OCRWorkerManager:
    """管理持久化 OCR 子进程，崩溃（含 SIGSEGV）后自动重启。"""

    OCR_TIMEOUT = 120

    def __init__(self, variant: str):
        self._variant = variant
        self._lock = threading.Lock()
        self._process: Optional[multiprocessing.Process] = None
        self._req_queue = None
        self._resp_queue = None
        self._start_worker()

    def _start_worker(self):
        ctx = multiprocessing.get_context('spawn')
        self._req_queue = ctx.Queue()
        self._resp_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_loop,
            args=(self._req_queue, self._resp_queue, self._variant),
            daemon=True,
        )
        self._process.start()
        logger.info(f"OCR 子进程已启动 (pid={self._process.pid})")

        try:
            status, payload = self._resp_queue.get(timeout=180)
            if status == 'init_error':
                raise RuntimeError(f"OCR 子进程初始化失败: {payload}")
            logger.info(f"OCR 子进程就绪 (pid={self._process.pid})")
        except queue.Empty:
            raise RuntimeError("OCR 子进程初始化超时（180s）")

    def _ensure_alive(self):
        if self._process is None or not self._process.is_alive():
            logger.warning(f"OCR 子进程已崩溃 (exit={self._process.exitcode if self._process else '?'})，重启中...")
            self._cleanup()
            self._start_worker()

    def _cleanup(self):
        if self._process and self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5)
        self._process = None

    def run_ocr(self, image_path: str):
        with self._lock:
            self._ensure_alive()
            self._req_queue.put(('ocr', image_path))
            try:
                status, payload = self._resp_queue.get(timeout=self.OCR_TIMEOUT)
            except queue.Empty:
                logger.error("OCR 响应超时，重启子进程")
                self._cleanup()
                self._start_worker()
                raise SystemOverloadError("OCR 推理超时")
            if status == 'ok':
                return payload
            detail = f"OCR 子进程错误: {payload}"
            logger.error(detail)
            if _is_overload_error(str(payload)):
                raise SystemOverloadError(detail)
            raise RuntimeError(detail)

    def warmup(self, image_path: str):
        with self._lock:
            self._ensure_alive()
            self._req_queue.put(('warmup', image_path))
            try:
                status, payload = self._resp_queue.get(timeout=180)
                if status != 'ok':
                    logger.warning(f"OCR 预热异常（可忽略）: {payload}")
            except queue.Empty:
                logger.warning("OCR 预热超时")

    def shutdown(self):
        with self._lock:
            if self._req_queue:
                try:
                    self._req_queue.put(None)
                except Exception:
                    pass
            self._cleanup()


# ── PaddleOCREngine ──────────────────────────────────────

@ocr_registry.register("paddle")
class PaddleOCREngine(BaseOCREngine):
    """封装 PaddleOCR，OCR 推理在子进程中执行以隔离 C++ 崩溃。"""

    MAX_OCR_SIDE = 960

    @staticmethod
    def _default_pool_size() -> int:
        from ...config import get_settings
        try:
            s = get_settings()
            if s.OCR_POOL_SIZE > 0:
                return s.OCR_POOL_SIZE
        except Exception:
            pass
        env_val = os.environ.get('OCR_POOL_SIZE', '')
        if env_val.isdigit() and int(env_val) > 0:
            return int(env_val)
        if os.environ.get('INSTALL_GPU', '').lower() == 'true':
            return 1
        return max(1, (os.cpu_count() or 2) - 1)

    def __init__(self, pool_size: int = 0):
        from ...config import get_settings
        try:
            variant = get_settings().OCR_MODEL
        except Exception:
            variant = os.environ.get('OCR_MODEL', 'server')
        n = pool_size or self._default_pool_size()
        gpu_mode = os.environ.get('INSTALL_GPU', '').lower() == 'true'
        logger.info(f"初始化 OCR 子进程池 (variant={variant}, gpu={gpu_mode}, workers={n})...")
        self._workers: list[_OCRWorkerManager] = []
        self._pool: queue.Queue = queue.Queue()
        for i in range(n):
            w = _OCRWorkerManager(variant)
            self._workers.append(w)
            self._pool.put(w)
            logger.info(f"OCR worker {i+1}/{n} 就绪")

    @staticmethod
    def _parse_poly(poly) -> Optional[list[list[int]]]:
        try:
            pts = poly if isinstance(poly, list) else list(poly)
            if len(pts) < 4:
                return None
            return [[int(round(p[0])), int(round(p[1]))] for p in pts]
        except Exception:
            return None

    def warmup(self) -> None:
        """预热所有子进程中的 OCR 实例。"""
        dummy_img = np.ones((64, 64, 3), dtype=np.uint8) * 255
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(tmp_fd)
        try:
            cv2.imwrite(tmp_path, dummy_img)
            for i, w in enumerate(self._workers):
                try:
                    w.warmup(tmp_path)
                except Exception as e:
                    logger.warning(f"预热 worker {i+1} 异常（可忽略）: {e}")
            logger.info(f"OCR 预热完成（{len(self._workers)} 个 worker）")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def recognize(self, image_path: str) -> list[TextLine]:
        """识别图片中的文本行，自动缩放超大图。"""
        started_at = time.time()
        decode_started_at = time.time()
        orig_img = cv2.imread(image_path)
        decode_ms = (time.time() - decode_started_at) * 1000
        scale = 1.0
        ocr_path = image_path
        resize_ms = 0.0
        encode_ms = 0.0
        original_size = (0, 0)
        used_size = (0, 0)

        if orig_img is not None:
            h, w = orig_img.shape[:2]
            original_size = (w, h)
            used_size = (w, h)
            max_side = max(h, w)
            if max_side > self.MAX_OCR_SIDE:
                resize_started_at = time.time()
                scale = self.MAX_OCR_SIDE / max_side
                new_w, new_h = int(w * scale), int(h * scale)
                small = cv2.resize(orig_img, (new_w, new_h),
                                   interpolation=cv2.INTER_AREA)
                resize_ms = (time.time() - resize_started_at) * 1000
                suffix = Path(image_path).suffix or '.jpg'
                tmp_fd, ocr_path = tempfile.mkstemp(suffix=suffix)
                os.close(tmp_fd)
                encode_started_at = time.time()
                cv2.imwrite(ocr_path, small, [cv2.IMWRITE_JPEG_QUALITY, 95])
                encode_ms = (time.time() - encode_started_at) * 1000
                used_size = (new_w, new_h)
                logger.info(f"图片缩放: {w}x{h} -> {new_w}x{new_h} (scale={scale:.3f})")

        try:
            infer_started_at = time.time()
            lines = self._run_ocr(ocr_path)
            infer_ms = (time.time() - infer_started_at) * 1000
        finally:
            if ocr_path != image_path and os.path.exists(ocr_path):
                os.unlink(ocr_path)

        if scale != 1.0 and lines:
            remap_started_at = time.time()
            inv = 1.0 / scale
            for line in lines:
                line.bbox = [
                    [int(p[0] * inv), int(p[1] * inv)]
                    for p in line.bbox
                ]
            remap_ms = (time.time() - remap_started_at) * 1000
        else:
            remap_ms = 0.0

        total_ms = (time.time() - started_at) * 1000
        _perf(
            "OCR recognize 完成: image=%s, original=%sx%s, used=%sx%s, lines=%s, decode_ms=%.1f, resize_ms=%.1f, encode_ms=%.1f, infer_ms=%.1f, remap_ms=%.1f, total_ms=%.1f"
            % (
                image_path,
                original_size[0], original_size[1],
                used_size[0], used_size[1],
                len(lines),
                decode_ms,
                resize_ms,
                encode_ms,
                infer_ms,
                remap_ms,
                total_ms,
            )
        )
        return lines

    def _run_ocr(self, image_path: str) -> list[TextLine]:
        """从 worker 池取一个子进程执行 OCR。"""
        worker: _OCRWorkerManager = self._pool.get()
        try:
            raw = worker.run_ocr(image_path)
        finally:
            self._pool.put(worker)

        if not raw:
            return []

        lines: list[TextLine] = []
        for i, item in enumerate(raw):
            api_type, text, poly, conf = item
            bbox = self._parse_poly(poly)
            if bbox is None:
                continue
            lines.append(TextLine(i, text, bbox, conf))

        logger.info(f"OCR 识别到 {len(lines)} 行文本")
        return lines