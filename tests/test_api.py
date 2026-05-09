"""Integration tests for the API routes using a mocked vision service."""
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.cache import cache
from app.main import app
from app.services.vision import vision_service


@pytest.fixture(autouse=True)
def _mock_model_loaded():
    """Pretend the model is loaded and clear cache for each test."""
    cache.clear()
    vision_service._loaded = True
    yield
    vision_service._loaded = False


MOCK_ATTRIBUTES = {
    "object_type": "warrior",
    "style": "Cyberpunk",
    "dominant_color": "Red",
    "mood": "Dark",
    "lighting": "Neon",
    "environment": "futuristic city",
}


def _make_test_image() -> bytes:
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestGenerateEndpoint:
    @patch.object(vision_service, "analyze", return_value=MOCK_ATTRIBUTES)
    def test_successful_generation(self, mock_analyze, client):
        image_bytes = _make_test_image()
        resp = client.post(
            "/api/v1/generate",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "image_hash" in data
        assert "metadata" in data
        meta = data["metadata"]
        assert "name" in meta
        assert "description" in meta
        assert isinstance(meta["attributes"], list)
        assert len(meta["attributes"]) >= 1
        mock_analyze.assert_called_once()

    def test_rejects_non_image(self, client):
        resp = client.post(
            "/api/v1/generate",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400

    def test_rejects_empty_file(self, client):
        resp = client.post(
            "/api/v1/generate",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    @patch.object(vision_service, "analyze", return_value=MOCK_ATTRIBUTES)
    def test_cache_hit_returns_same_result(self, mock_analyze, client):
        image_bytes = _make_test_image()
        resp1 = client.post("/api/v1/generate", files={"file": ("t.png", image_bytes, "image/png")})
        resp2 = client.post("/api/v1/generate", files={"file": ("t.png", image_bytes, "image/png")})
        assert resp1.json() == resp2.json()
        # Model called only once due to cache
        assert mock_analyze.call_count == 1

    @patch.object(vision_service, "analyze", return_value=MOCK_ATTRIBUTES)
    def test_response_schema_matches_erc721(self, mock_analyze, client):
        image_bytes = _make_test_image()
        resp = client.post("/api/v1/generate", files={"file": ("t.png", image_bytes, "image/png")})
        data = resp.json()
        # ERC-721 required fields
        meta = data["metadata"]
        assert isinstance(meta["name"], str)
        assert isinstance(meta["description"], str)
        for attr in meta["attributes"]:
            assert "trait_type" in attr
            assert "value" in attr
