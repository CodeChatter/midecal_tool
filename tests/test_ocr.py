import importlib
import os
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reset_runtime_module():
    import app.bootstrap.runtime as runtime

    runtime._BOOTSTRAPPED = False
    return runtime


def _clear_runtime_env():
    os.environ.pop("FLAGS_enable_pir_api", None)
    os.environ.pop("FLAGS_enable_pir_in_executor", None)


def test_bootstrap_runtime_sets_required_env_vars():
    runtime = _reset_runtime_module()

    _clear_runtime_env()

    runtime.bootstrap_runtime()

    assert os.environ["FLAGS_enable_pir_api"] == "0"
    assert os.environ["FLAGS_enable_pir_in_executor"] == "0"


def test_bootstrap_runtime_is_idempotent_and_preserves_existing_values():
    runtime = _reset_runtime_module()

    os.environ["FLAGS_enable_pir_api"] = "1"
    os.environ["FLAGS_enable_pir_in_executor"] = "1"

    runtime.bootstrap_runtime()
    runtime.bootstrap_runtime()

    assert os.environ["FLAGS_enable_pir_api"] == "1"
    assert os.environ["FLAGS_enable_pir_in_executor"] == "1"


def test_import_app_package_does_not_bootstrap_runtime():
    runtime = _reset_runtime_module()

    _clear_runtime_env()

    sys.modules.pop("app", None)
    importlib.import_module("app")

    assert os.environ.get("FLAGS_enable_pir_api") is None
    assert os.environ.get("FLAGS_enable_pir_in_executor") is None
    assert runtime._BOOTSTRAPPED is False


def test_import_app_main_bootstraps_runtime():
    import app.bootstrap.runtime as runtime

    runtime._BOOTSTRAPPED = False
    _clear_runtime_env()

    sys.modules.pop("app.main", None)
    importlib.import_module("app.main")

    assert os.environ["FLAGS_enable_pir_api"] == "0"
    assert os.environ["FLAGS_enable_pir_in_executor"] == "0"


def test_create_ocr_instance_bootstraps_before_paddle_import():
    import app.bootstrap.runtime as runtime
    from app.core.ocr.paddle import PaddleOCREngine

    runtime._BOOTSTRAPPED = False
    _clear_runtime_env()

    captured = {}

    class FakePaddleOCR:
        def __init__(self, *args, **kwargs):
            captured["pir_api"] = os.environ.get("FLAGS_enable_pir_api")
            captured["pir_executor"] = os.environ.get("FLAGS_enable_pir_in_executor")
            captured["enable_mkldnn"] = kwargs.get("enable_mkldnn")
            captured["enable_hpi"] = kwargs.get("enable_hpi")

    fake_module = type("FakePaddleOCRModule", (), {"PaddleOCR": FakePaddleOCR})

    with mock.patch.dict(sys.modules, {"paddleocr": fake_module}):
        PaddleOCREngine._create_ocr_instance()

    assert captured == {
        "pir_api": "0",
        "pir_executor": "0",
        "enable_mkldnn": False,
        "enable_hpi": False,
    }


def test_run_ocr_impl_falls_back_to_legacy_ocr_without_cls_arg():
    from app.core.ocr.paddle import PaddleOCREngine

    class HybridOCR:
        def predict(self, image_path):
            raise RuntimeError("ConvertPirAttribute2RuntimeAttribute not support")

        def ocr(self, *args, **kwargs):
            if "cls" in kwargs:
                raise TypeError("PaddleOCR.predict() got an unexpected keyword argument 'cls'")
            return [[
                [
                    [[0, 0], [10, 0], [10, 10], [0, 10]],
                    ("李四", 0.88),
                ]
            ]]

    engine = object.__new__(PaddleOCREngine)
    lines = engine._run_ocr_impl(HybridOCR(), "dummy.png")

    assert len(lines) == 1
    assert lines[0].text == "李四"
    assert lines[0].confidence == 0.88


def test_warmup_calls_run_ocr_impl_for_each_pool_instance():
    """warmup() 应对池中每个实例调用一次 _run_ocr_impl。"""
    import queue
    from app.core.ocr.paddle import PaddleOCREngine

    class FakeOCR:
        def predict(self, image_path):
            return [{'rec_texts': [], 'dt_polys': []}]

    engine = object.__new__(PaddleOCREngine)
    engine._pool = queue.Queue()
    fake1, fake2 = FakeOCR(), FakeOCR()
    engine._pool.put(fake1)
    engine._pool.put(fake2)

    engine.warmup()

    # 两个实例都应该被归还到池中
    assert engine._pool.qsize() == 2


def test_warmup_tolerates_ocr_exception():
    """warmup() 中某个实例推理失败不应抛出异常，实例仍归还到池中。"""
    import queue
    from app.core.ocr.paddle import PaddleOCREngine

    class CrashingOCR:
        def predict(self, image_path):
            raise RuntimeError("模拟推理失败")

        def ocr(self, *args, **kwargs):
            raise RuntimeError("模拟推理失败")

    engine = object.__new__(PaddleOCREngine)
    engine._pool = queue.Queue()
    engine._pool.put(CrashingOCR())

    # 不应抛出异常
    engine.warmup()

    # 实例应归还到池中
    assert engine._pool.qsize() == 1


def test_prewarm_ocr_engine_calls_warmup():
    """prewarm_ocr_engine 应在创建引擎后调用 warmup()。"""
    from app.api.v1.deps import prewarm_ocr_engine, _ocr_engines

    warmup_called = False

    class FakeEngine:
        def warmup(self):
            nonlocal warmup_called
            warmup_called = True

    # 清除缓存，注入 fake
    _ocr_engines.pop("test_engine", None)
    with mock.patch("app.api.v1.deps.get_ocr_engine", return_value=FakeEngine()):
        prewarm_ocr_engine("test_engine")

    assert warmup_called
    _ocr_engines.pop("test_engine", None)
