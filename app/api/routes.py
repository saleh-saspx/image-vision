import logging
import time

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from app.core.cache import cache
from app.core.jobs import job_store
from app.models.schemas import (
    HealthResponse,
    JobAccepted,
    JobResult,
    JobStatus,
    NFTResponse,
    QueueStats,
)
from app.services.pipeline import process_image
from app.services.vision import vision_service
from app.services.worker import inference_worker
from app.utils.hash import compute_sha256

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def _read_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use JPEG, PNG or WebP.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size is {MAX_FILE_SIZE // (1024*1024)}MB.")
    if len(raw_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")

    return raw_bytes


@router.post(
    "/generate",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue an image for metadata generation",
)
async def generate_nft_metadata(response: Response, file: UploadFile = File(...)):
    """Hash the upload, queue it, and return the id straight away.

    The client never waits on inference here, so a slow CPU can no longer turn
    into a gateway timeout. Poll `poll_url` for the result.
    """
    raw_bytes = await _read_upload(file)
    image_hash = compute_sha256(raw_bytes)
    tag = image_hash[:12]
    poll_url = f"/api/v1/jobs/{image_hash}"

    # Already generated once — hand back a finished job without touching the queue.
    cached = cache.get(image_hash)
    if cached is not None:
        job_store.complete_now(image_hash, cached)
        logger.info("[%s] Cache hit on upload — job returned complete", tag)
        response.status_code = status.HTTP_200_OK
        return JobAccepted(
            job_id=image_hash,
            image_hash=image_hash,
            status=JobStatus.COMPLETED,
            poll_url=poll_url,
            deduplicated=True,
        )

    job, created = job_store.create_or_get(image_hash)

    if not created:
        # Same image already queued or running: join it rather than duplicate work.
        logger.info("[%s] Joined existing job (status=%s)", tag, job.status.value)
        if job.status == JobStatus.COMPLETED:
            response.status_code = status.HTTP_200_OK
        return JobAccepted(
            job_id=job.id,
            image_hash=image_hash,
            status=job.status,
            poll_url=poll_url,
            queue_position=job_store.queue_position(job.id),
            deduplicated=True,
        )

    if not inference_worker.submit(image_hash, raw_bytes):
        # Never leave a job stranded in QUEUED that no worker will ever see.
        job_store.discard(image_hash)
        raise HTTPException(
            503,
            f"Inference queue is full ({inference_worker.capacity} waiting). Retry shortly.",
        )

    logger.info("[%s] Queued (depth=%d)", tag, inference_worker.depth)
    return JobAccepted(
        job_id=image_hash,
        image_hash=image_hash,
        status=JobStatus.QUEUED,
        poll_url=poll_url,
        queue_position=job_store.queue_position(image_hash),
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResult,
    summary="Fetch a queued job's status or result",
)
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            404,
            "Unknown job id. It may have expired, or the upload was never queued.",
        )

    return JobResult(
        job_id=job.id,
        image_hash=job.id,
        status=job.status,
        metadata=job.result.metadata if job.result else None,
        error=job.error,
        queue_position=job_store.queue_position(job.id),
        waited_ms=job.waited_ms,
        duration_ms=job.duration_ms,
        cached=bool(job.result and job.result.cached),
    )


@router.post(
    "/generate/sync",
    response_model=NFTResponse,
    summary="Generate metadata inline (blocks until done)",
)
async def generate_nft_metadata_sync(file: UploadFile = File(...)):
    """Original blocking behaviour, kept for small images and local testing.

    Bypasses the queue entirely, so it is still subject to gateway timeouts on
    slow hardware — prefer POST /generate.
    """
    raw_bytes = await _read_upload(file)

    t0 = time.monotonic()
    result = await process_image(raw_bytes)
    logger.info("Sync request completed in %.2fs", time.monotonic() - t0)
    return result


@router.get("/queue", response_model=QueueStats, summary="Queue depth and job counts")
async def queue_stats():
    stats = job_store.stats()
    return QueueStats(
        queued=stats["queued"],
        processing=stats["processing"],
        completed=stats["completed"],
        failed=stats["failed"],
        total=stats["total"],
        workers=inference_worker.worker_count,
        capacity=inference_worker.capacity,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", model_loaded=vision_service.is_loaded)


@router.get("/debug")
async def debug_info():
    """Quick diagnostics — no image needed."""
    import torch

    return {
        "model_loaded": vision_service.is_loaded,
        "device": str(vision_service._device) if vision_service._device else "not set",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cache_size": cache.size,
        "queue_depth": inference_worker.depth,
        "queue_running": inference_worker.is_running,
        "jobs": job_store.stats(),
    }
