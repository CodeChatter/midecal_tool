"""运行时启动初始化。"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAPPED = False

_PADDLE_ENV_DEFAULTS = {
    "FLAGS_enable_pir_api": "0",
    "FLAGS_enable_pir_in_executor": "0",
    "FLAGS_use_mkldnn": "0",
    "FLAGS_allocator_strategy": "naive_best_fit",
    "FLAGS_fraction_of_gpu_memory_to_use": "0.6",
    "FLAGS_num_threads": "1",
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def bootstrap_runtime() -> None:
    """初始化运行时环境变量。可安全重复调用。"""
    global _BOOTSTRAPPED

    if _BOOTSTRAPPED:
        return

    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAPPED:
            return

        applied = {}
        for key, value in _PADDLE_ENV_DEFAULTS.items():
            os.environ.setdefault(key, value)
            applied[key] = os.environ.get(key, value)

        logger.info(
            "运行时初始化完成: %s",
            ", ".join(f"{k}={v}" for k, v in applied.items()),
        )
        _BOOTSTRAPPED = True
