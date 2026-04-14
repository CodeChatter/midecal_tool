from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient


def test_mask_request_default_mode_is_white():
    from app.schemas.mask import MaskRequest

    req = MaskRequest(image_url="https://example.com/test.png")

    assert req.mask_mode == "white"


def test_startup_and_first_ocr_request_succeeds_without_crash():
    import app.api.v1.deps as deps
    import app.api.v1.endpoints.ocr as ocr_endpoint
    import app.main as main

    class FakeEngine:
        def warmup(self):
            return None

        def recognize(self, image_path):
            return []

    fake_engine = FakeEngine()

    def fake_download(loc, local_path):
        Path(local_path).write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x04\x00\x01"
            b"\x0b\xe7\x02\x9d\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return local_path

    with mock.patch.object(deps, "get_ocr_engine", return_value=fake_engine), \
            mock.patch.object(deps, "get_masker", return_value=object()), \
            mock.patch.object(main, "prewarm_ocr_engine", return_value=fake_engine), \
            mock.patch.object(main, "get_masker", return_value=object()), \
            mock.patch.object(ocr_endpoint, "get_ocr_engine", return_value=fake_engine), \
            mock.patch.object(ocr_endpoint.cos, "parse_url", return_value=mock.Mock(key="test.png")), \
            mock.patch.object(ocr_endpoint.cos, "download_to_local", side_effect=fake_download):
        deps._ocr_engines.clear()
        deps._maskers.clear()
        client = TestClient(main.app)

        with client:
            response = client.post(
                "/api/ocr",
                json={"image_url": "https://example-1250000000.cos.ap-shanghai.myqcloud.com/test/form/test.png"},
            )

        assert response.status_code == 200
        assert response.json() == {"lines": [], "full_text": "", "total_lines": 0}


def test_mask_request_returns_expected_success_payload():
    import app.api.v1.deps as deps
    import app.api.v1.endpoints.mask as mask_endpoint
    import app.main as main

    class FakeEngine:
        def warmup(self):
            return None

    class FakeMasker:
        def process_image(self, input_path, output_path, categories):
            Path(output_path).write_bytes(Path(input_path).read_bytes())

    fake_engine = FakeEngine()
    fake_masker = FakeMasker()
    image_url = "https://example-1250000000.cos.ap-shanghai.myqcloud.com/test/form/test.png"
    masked_url = "https://example-1250000000.cos.ap-shanghai.myqcloud.com/test/form/test_masked.png"

    def fake_download(loc, local_path):
        Path(local_path).write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x04\x00\x01"
            b"\x0b\xe7\x02\x9d\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return local_path

    loc = mock.Mock(bucket="example-1250000000", region="ap-shanghai", key="test/form/test.png")

    with mock.patch.object(deps, "get_ocr_engine", return_value=fake_engine), \
            mock.patch.object(deps, "get_masker", return_value=fake_masker), \
            mock.patch.object(main, "prewarm_ocr_engine", return_value=fake_engine), \
            mock.patch.object(main, "get_masker", return_value=fake_masker), \
            mock.patch.object(mask_endpoint, "get_masker", return_value=fake_masker), \
            mock.patch.object(mask_endpoint.cos, "parse_url", return_value=loc), \
            mock.patch.object(mask_endpoint.cos, "download_to_local", side_effect=fake_download), \
            mock.patch.object(mask_endpoint.cos, "upload_file", return_value=masked_url):
        deps._ocr_engines.clear()
        deps._maskers.clear()
        client = TestClient(main.app)

        with client:
            response = client.post(
                "/api/mask",
                json={"image_url": image_url, "llm_provider": "openai", "mask_mode": "black"},
            )

        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "origin_url": image_url,
            "masked_url": masked_url,
            "error": None,
        }


def test_mask_request_accepts_gray_mode():
    import app.api.v1.deps as deps
    import app.api.v1.endpoints.mask as mask_endpoint
    import app.main as main

    class FakeEngine:
        def warmup(self):
            return None

    class FakeMasker:
        def process_image(self, input_path, output_path, categories):
            Path(output_path).write_bytes(Path(input_path).read_bytes())

    fake_engine = FakeEngine()
    fake_masker = FakeMasker()
    image_url = "https://example-1250000000.cos.ap-shanghai.myqcloud.com/test/form/test.png"
    masked_url = "https://example-1250000000.cos.ap-shanghai.myqcloud.com/test/form/test_masked_gray.png"

    def fake_download(loc, local_path):
        Path(local_path).write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x04\x00\x01"
            b"\x0b\xe7\x02\x9d\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return local_path

    loc = mock.Mock(bucket="example-1250000000", region="ap-shanghai", key="test/form/test.png")

    with mock.patch.object(deps, "get_ocr_engine", return_value=fake_engine), \
            mock.patch.object(deps, "get_masker", return_value=fake_masker), \
            mock.patch.object(main, "prewarm_ocr_engine", return_value=fake_engine), \
            mock.patch.object(main, "get_masker", return_value=fake_masker), \
            mock.patch.object(mask_endpoint, "get_masker", return_value=fake_masker), \
            mock.patch.object(mask_endpoint.cos, "parse_url", return_value=loc), \
            mock.patch.object(mask_endpoint.cos, "download_to_local", side_effect=fake_download), \
            mock.patch.object(mask_endpoint.cos, "upload_file", return_value=masked_url):
        deps._ocr_engines.clear()
        deps._maskers.clear()
        client = TestClient(main.app)

        with client:
            response = client.post(
                "/api/mask",
                json={"image_url": image_url, "llm_provider": "openai", "mask_mode": "gray"},
            )

        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "origin_url": image_url,
            "masked_url": masked_url,
            "error": None,
        }


def test_mask_request_returns_expected_error_payload_for_invalid_mode():
    import app.api.v1.deps as deps
    import app.main as main
    from app.core.registry import masking_registry

    image_url = "https://example-1250000000.cos.ap-shanghai.myqcloud.com/test/form/test.png"

    class FakeEngine:
        def warmup(self):
            return None

    with mock.patch.object(deps, "get_ocr_engine", return_value=FakeEngine()), \
            mock.patch.object(deps, "get_masker", return_value=object()), \
            mock.patch.object(main, "prewarm_ocr_engine", return_value=FakeEngine()), \
            mock.patch.object(main, "get_masker", return_value=object()):
        deps._ocr_engines.clear()
        deps._maskers.clear()
        client = TestClient(main.app)

        with client:
            response = client.post(
                "/api/mask",
                json={
                    "image_url": image_url,
                    "mask_mode": "invalid-mode",
                },
            )

        assert response.status_code == 200
        assert response.json() == {
            "success": False,
            "origin_url": image_url,
            "masked_url": None,
            "error": f"无效的 mask_mode，可选: {masking_registry.available()}",
        }
