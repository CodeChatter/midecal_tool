"""人脸检测器：给 Portrait 类敏感区域产出精确 bbox。

默认优先使用 YuNet（OpenCV 4.5+ 自带接口，精度优于 Haar），
若对应 ONNX 模型文件未就位则回退 Haar 级联分类器（OpenCV 安装时自带）。
两者均为 CPU 推理，不占 GPU 显存。
"""

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 模型文件默认位置：项目根下 models/face_detection_yunet.onnx
# 若环境变量 FACE_YUNET_MODEL 存在则覆盖
_YUNET_ENV = "FACE_YUNET_MODEL"
_YUNET_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet.onnx"


class FaceDetector:
    """单例式人脸检测器（首次使用时延迟初始化）。"""

    _instance: Optional["FaceDetector"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_backend()
        return cls._instance

    def _init_backend(self) -> None:
        self._backend: Optional[str] = None
        self._yunet = None
        self._haar: Optional[cv2.CascadeClassifier] = None

        # 1) 尝试 YuNet
        model_path = os.environ.get(_YUNET_ENV) or str(_YUNET_DEFAULT_PATH)
        if hasattr(cv2, "FaceDetectorYN_create") and Path(model_path).exists():
            try:
                self._yunet = cv2.FaceDetectorYN_create(
                    model_path, "", (320, 320), 0.6, 0.3, 5000
                )
                self._backend = "yunet"
                logger.info(f"人脸检测器就绪: backend=YuNet, model={model_path}")
                return
            except Exception as e:
                logger.warning(f"YuNet 初始化失败，将回退 Haar: {e}")

        # 2) 回退 Haar
        haar_xml = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        try:
            self._haar = cv2.CascadeClassifier(haar_xml)
            if self._haar.empty():
                raise RuntimeError("Haar 分类器加载为空")
            self._backend = "haar"
            logger.info(
                f"人脸检测器就绪: backend=Haar, "
                f"未发现 YuNet 模型，精度较低；"
                f"建议下载 face_detection_yunet_2023mar.onnx 到 models/ 以启用 YuNet"
            )
        except Exception as e:
            logger.error(f"Haar 初始化失败: {e}")
            self._backend = None

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def detect(self, image: np.ndarray, min_size_ratio: float = 0.02
               ) -> List[Tuple[int, int, int, int]]:
        """对 BGR 图像做人脸检测，返回 [(x1, y1, x2, y2), ...]。

        min_size_ratio: 过滤短边占图片短边比例小于该值的人脸框，
                        默认 0.02（500 像素短边上过滤 <10 像素的噪声框）。
        """
        if image is None or image.size == 0 or self._backend is None:
            return []

        h, w = image.shape[:2]
        short_side = min(h, w)
        min_face = max(16, int(short_side * min_size_ratio))

        if self._backend == "yunet":
            return self._detect_yunet(image, min_face)
        if self._backend == "haar":
            return self._detect_haar(image, min_face)
        return []

    def _detect_yunet(self, image: np.ndarray, min_face: int
                      ) -> List[Tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        assert self._yunet is not None
        self._yunet.setInputSize((w, h))
        try:
            _, faces = self._yunet.detect(image)
        except Exception as e:
            logger.warning(f"YuNet 推理失败: {e}")
            return []
        if faces is None:
            return []
        out = []
        for row in faces:
            x, y, fw, fh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if fw < min_face or fh < min_face:
                continue
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + fw)
            y2 = min(h, y + fh)
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
        return out

    def _detect_haar(self, image: np.ndarray, min_face: int
                     ) -> List[Tuple[int, int, int, int]]:
        assert self._haar is not None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(min_face, min_face),
        )
        h, w = image.shape[:2]
        out = []
        for (x, y, fw, fh) in faces:
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + fw))
            y2 = min(h, int(y + fh))
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
        return out
