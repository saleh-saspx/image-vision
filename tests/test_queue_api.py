"""End-to-end tests for the queued upload -> poll flow."""
import io
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.core.cache import cache
from app.core.jobs import job_store
from app.main import app
from app.services.vision import vision_service
from app.services.worker import inference_worker

MOCK_FIELDS = {
    "primary_subject": "a cyber skull",
    "objects": ["Skull", "Wires"],
    "scene": "street",
    "environment": "outdoor",
    "style": "cyberpunk",
    "art_medium": "3d render",
    "materials": ["metal"],
    "lighting": "neon lights",
    "mood": "dark",
}


@pytest.fixture(autouse=True)
def _clean_state():
    cache.clear()
    job_store.clear()
    vision_service._loaded = True
    yield
    vision_service._loaded = False
    job_store.clear()
    cache.clear()


@pytest.fixture()
def worker():
    """Run the real queue — lifespan is not triggered by TestClient here, so the
    worker must be started explicitly (and never loads the actual model)."""
    inference_worker.start()
    yield inference_worker
    inference_worker.stop(timeout=5)


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _image(color=(46, 96, 200), size=(160, 120)) -> bytes:
    img = Image.new("RGB", size, color=color)
    ImageDraw.Draw(img).rectangle([0, 0, size[0] // 2, size[1]], fill=(212, 170, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, image_bytes, name="t.png"):
    return client.post("/api/v1/generate", files={"file": (name, image_bytes, "image/png")})


def _poll(client, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/api/v1/jobs/{job_id}").json()
        if payload["status"] in ("completed", "failed"):
            return payload
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


class TestSubmission:
    def test_upload_returns_immediately_with_a_hash(self, client, worker):
        resp = _upload(client, _image())
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "queued"
        assert len(body["job_id"]) == 64  # sha256 hex
        assert body["job_id"] == body["image_hash"]
        assert body["poll_url"] == f"/api/v1/jobs/{body['job_id']}"

    def test_job_id_is_the_image_hash(self, client, worker):
        from app.utils.hash import compute_sha256

        image_bytes = _image()
        body = _upload(client, image_bytes).json()
        assert body["job_id"] == compute_sha256(image_bytes)

    def test_identical_uploads_share_one_job(self, client, worker):
        image_bytes = _image()
        with patch.object(vision_service, "analyze", return_value=MOCK_FIELDS) as mock:
            first = _upload(client, image_bytes).json()
            second = _upload(client, image_bytes).json()
            assert first["job_id"] == second["job_id"]
            assert second["deduplicated"] is True

            _poll(client, first["job_id"])
            # Deduplicated upload must not have queued a second inference.
            assert mock.call_count == 1

    def test_different_images_get_different_jobs(self, client, worker):
        first = _upload(client, _image(color=(200, 46, 46))).json()
        second = _upload(client, _image(color=(58, 150, 70))).json()
        assert first["job_id"] != second["job_id"]

    def test_validation_happens_before_queueing(self, client, worker):
        resp = client.post("/api/v1/generate", files={"file": ("x.txt", b"nope", "text/plain")})
        assert resp.status_code == 400
        assert job_store.stats()["total"] == 0

    def test_empty_upload_rejected(self, client, worker):
        assert client.post(
            "/api/v1/generate", files={"file": ("e.png", b"", "image/png")}
        ).status_code == 400


class TestPolling:
    def test_result_is_retrievable_by_hash(self, client, worker):
        with patch.object(vision_service, "analyze", return_value=MOCK_FIELDS):
            job_id = _upload(client, _image()).json()["job_id"]
            result = _poll(client, job_id)

        assert result["status"] == "completed"
        assert result["job_id"] == job_id
        meta = result["metadata"]
        assert meta["title"]
        assert meta["style"] == "Cyberpunk"
        assert len(meta["attributes"]) > 6
        assert meta["confidence"]

    def test_timings_are_reported(self, client, worker):
        with patch.object(vision_service, "analyze", return_value=MOCK_FIELDS):
            job_id = _upload(client, _image()).json()["job_id"]
            result = _poll(client, job_id)

        assert result["duration_ms"] is not None
        assert result["waited_ms"] is not None

    def test_unknown_job_returns_404(self, client, worker):
        assert client.get("/api/v1/jobs/" + "0" * 64).status_code == 404

    def test_failed_job_reports_error_not_500(self, client, worker):
        with patch.object(vision_service, "analyze", side_effect=ValueError("bad json")):
            job_id = _upload(client, _image()).json()["job_id"]
            result = _poll(client, job_id)

        # The poll itself succeeds; the failure is carried in the payload.
        assert result["status"] == "failed"
        assert "ValueError" in result["error"]
        assert result["metadata"] is None

    def test_worker_survives_a_failed_job(self, client, worker):
        with patch.object(vision_service, "analyze", side_effect=ValueError("boom")):
            bad = _upload(client, _image(color=(10, 10, 10))).json()["job_id"]
            _poll(client, bad)

        with patch.object(vision_service, "analyze", return_value=MOCK_FIELDS):
            good = _upload(client, _image(color=(200, 46, 46))).json()["job_id"]
            assert _poll(client, good)["status"] == "completed"


class TestCacheInteraction:
    def test_second_upload_of_finished_image_returns_completed_at_once(self, client, worker):
        image_bytes = _image()
        with patch.object(vision_service, "analyze", return_value=MOCK_FIELDS) as mock:
            job_id = _upload(client, image_bytes).json()["job_id"]
            _poll(client, job_id)

            repeat = _upload(client, image_bytes)
            assert repeat.status_code == 200
            assert repeat.json()["status"] == "completed"
            assert repeat.json()["deduplicated"] is True
            assert mock.call_count == 1

    def test_cached_result_matches_original(self, client, worker):
        image_bytes = _image()
        with patch.object(vision_service, "analyze", return_value=MOCK_FIELDS):
            job_id = _upload(client, image_bytes).json()["job_id"]
            first = _poll(client, job_id)
            _upload(client, image_bytes)
            second = client.get(f"/api/v1/jobs/{job_id}").json()

        assert first["metadata"] == second["metadata"]


class TestBackpressure:
    def test_full_queue_returns_503_and_leaves_no_ghost_job(self, client):
        # No worker running, so nothing drains: the queue fills and rejects.
        with patch.object(inference_worker, "submit", return_value=False):
            resp = _upload(client, _image())

        assert resp.status_code == 503
        assert "queue is full" in resp.json()["detail"].lower()
        # A rejected upload must not leave a job stuck in QUEUED forever.
        assert job_store.stats()["total"] == 0


class TestQueueStats:
    def test_stats_endpoint_reports_counts(self, client, worker):
        with patch.object(vision_service, "analyze", return_value=MOCK_FIELDS):
            job_id = _upload(client, _image()).json()["job_id"]
            _poll(client, job_id)

        stats = client.get("/api/v1/queue").json()
        assert stats["completed"] == 1
        assert stats["workers"] >= 1
        assert stats["capacity"] >= 1


class TestSyncEndpointStillWorks:
    def test_sync_path_returns_metadata_inline(self, client):
        with patch.object(vision_service, "analyze", return_value=MOCK_FIELDS):
            resp = client.post(
                "/api/v1/generate/sync",
                files={"file": ("t.png", _image(), "image/png")},
            )
        assert resp.status_code == 200
        assert resp.json()["metadata"]["title"]
