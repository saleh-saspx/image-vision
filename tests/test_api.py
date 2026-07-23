"""Integration tests for the API routes using a mocked vision service."""
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

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


# What VisionService.analyze returns after key mapping.
MOCK_FIELDS = {
    "primary_subject": "a cyber skull",
    "secondary_subjects": ["circuitry"],
    "objects": ["Skull", "Wires", "Neon Sign"],
    "scene": "street",
    "environment": "outdoor",
    "style": "cyberpunk",
    "art_medium": "3d render",
    "materials": ["metal"],
    "lighting": "neon lights",
    "mood": "dark",
    "perspective": "front",
    "texture": "glossy",
    "pattern": "geometric",
}


def _make_test_image() -> bytes:
    img = Image.new("RGB", (160, 120), color=(46, 96, 200))
    ImageDraw.Draw(img).rectangle([0, 0, 79, 119], fill=(212, 170, 60))
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
    @patch.object(vision_service, "analyze", return_value=MOCK_FIELDS)
    def test_successful_generation(self, mock_analyze, client):
        resp = client.post(
            "/api/v1/generate/sync",
            files={"file": ("test.png", _make_test_image(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "image_hash" in data
        meta = data["metadata"]

        assert meta["title"] and "#" not in meta["title"]
        assert meta["name"] == meta["title"]
        assert meta["description"].endswith(".")
        assert meta["category"] is not None
        assert meta["style"] == "Cyberpunk"
        assert meta["dominant_colors"]
        assert meta["visual_complexity"] in {"Simple", "Medium", "Complex", "Highly Detailed"}
        assert 15 <= len(meta["tags"]) <= 30
        assert len(meta["attributes"]) > 6
        assert meta["confidence"]
        mock_analyze.assert_called_once()

    @patch.object(vision_service, "analyze", return_value=MOCK_FIELDS)
    def test_response_contains_every_spec_field(self, mock_analyze, client):
        resp = client.post("/api/v1/generate/sync", files={"file": ("t.png", _make_test_image(), "image/png")})
        meta = resp.json()["metadata"]
        for field in (
            "title", "description", "category", "style", "primary_subject",
            "secondary_subjects", "scene", "environment", "objects", "materials",
            "dominant_colors", "lighting", "mood", "composition", "perspective",
            "texture", "pattern", "visual_complexity", "art_medium", "tags",
            "attributes", "confidence",
        ):
            assert field in meta, f"missing field {field}"

    @patch.object(vision_service, "analyze", return_value=MOCK_FIELDS)
    def test_confidence_is_capped(self, mock_analyze, client):
        resp = client.post("/api/v1/generate/sync", files={"file": ("t.png", _make_test_image(), "image/png")})
        confidence = resp.json()["metadata"]["confidence"]
        assert confidence
        assert all(0 < v <= 0.9 for v in confidence.values())

    @patch.object(vision_service, "analyze", return_value={})
    def test_degrades_gracefully_when_model_returns_nothing(self, mock_analyze, client):
        """Pixel-derived metadata must survive a useless model response."""
        resp = client.post("/api/v1/generate/sync", files={"file": ("t.png", _make_test_image(), "image/png")})
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert meta["title"]
        assert meta["dominant_colors"]
        assert meta["visual_complexity"]

    @patch.object(vision_service, "analyze", side_effect=ValueError("bad json"))
    def test_model_failure_returns_500(self, mock_analyze, client):
        resp = client.post("/api/v1/generate/sync", files={"file": ("t.png", _make_test_image(), "image/png")})
        assert resp.status_code == 500

    def test_rejects_non_image(self, client):
        resp = client.post(
            "/api/v1/generate/sync",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400

    def test_rejects_empty_file(self, client):
        resp = client.post(
            "/api/v1/generate/sync",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    @patch.object(vision_service, "analyze", return_value=MOCK_FIELDS)
    def test_cache_hit_returns_same_metadata(self, mock_analyze, client):
        image_bytes = _make_test_image()
        first = client.post("/api/v1/generate/sync", files={"file": ("t.png", image_bytes, "image/png")}).json()
        second = client.post("/api/v1/generate/sync", files={"file": ("t.png", image_bytes, "image/png")}).json()

        assert first["metadata"] == second["metadata"]
        assert first["cached"] is False
        assert second["cached"] is True
        assert mock_analyze.call_count == 1

    @patch.object(vision_service, "analyze", return_value=MOCK_FIELDS)
    def test_response_schema_matches_erc721(self, mock_analyze, client):
        resp = client.post("/api/v1/generate/sync", files={"file": ("t.png", _make_test_image(), "image/png")})
        meta = resp.json()["metadata"]
        assert isinstance(meta["name"], str)
        assert isinstance(meta["description"], str)
        for attr in meta["attributes"]:
            assert "trait_type" in attr
            assert "value" in attr
