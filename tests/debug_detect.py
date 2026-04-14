"""对比测试：zhipu vs qwen3-openai on test5/test6"""
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / '.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from app.core.ocr.paddle import PaddleOCREngine
from app.core.pipeline import MedicalPrivacyMasker
from app.core.masking.solid import SolidColorStrategy
from app.core.detectors.openai import OpenAIDetector
from app.core.detectors.zhipu import ZhipuDetector

out_dir = Path(__file__).parent / 'output'
out_dir.mkdir(parents=True, exist_ok=True)

images = ['test5.jpg', 'test6.jpeg']
providers = {
    'qwen3-openai': OpenAIDetector,
    'zhipu': ZhipuDetector,
}

for provider_name, detector_cls in providers.items():
    print(f"\n{'='*70}")
    print(f"Provider: {provider_name}")
    print(f"{'='*70}")

    try:
        pipeline = MedicalPrivacyMasker(
            ocr=PaddleOCREngine(pool_size=1),
            detector=detector_cls(),
            mask_strategy=SolidColorStrategy(color=(0, 0, 0)),
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        continue

    for name in images:
        image_path = str(Path(__file__).parent / 'input' / name)
        out_path = str(out_dir / f"{Path(name).stem}_{provider_name}_masked{Path(name).suffix}")
        t0 = time.time()
        try:
            stats = pipeline.process_image(image_path, out_path)
            print(f"{name}: {time.time()-t0:.1f}s | {stats.sensitive_count} 处 → {stats.sensitive_items}")
        except Exception as e:
            print(f"{name}: 失败 → {e}")
