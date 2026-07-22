import asyncio
import logging
import time
from functools import partial

from fastapi import HTTPException

from app.core.cache import cache
from app.models.schemas import NFTResponse
from app.services.imagestats import analyze_pixels
from app.services.nft_generator import generate_metadata
from app.services.vision import vision_service
from app.utils.hash import compute_sha256
from app.utils.image import preprocess_image

logger = logging.getLogger(__name__)

INFERENCE_TIMEOUT = 120  # seconds


def _blocking_inference(raw_bytes: bytes, image_hash: str) -> NFTResponse:
    """CPU-bound work that must NOT run on the event loop."""
    t0 = time.monotonic()
    tag = image_hash[:12]

    # Decode once; both the pixel analysis and the model read this same object.
    image = preprocess_image(raw_bytes)
    t1 = time.monotonic()

    # Measured facts first — a few milliseconds, and they are independent of
    # whether the model produces anything usable.
    stats = analyze_pixels(image)
    t2 = time.monotonic()
    logger.info(
        "[%s] Preprocess %.2fs, pixel stats %.3fs (colors=%s, complexity=%s)",
        tag, t1 - t0, t2 - t1, stats.colors, stats.complexity,
    )

    raw_fields = vision_service.analyze(image)
    t3 = time.monotonic()
    logger.info("[%s] Inference %.2fs. Fields: %s", tag, t3 - t2, raw_fields)

    metadata = generate_metadata(raw_fields, stats)
    total = time.monotonic() - t0
    logger.info("[%s] Done in %.2fs — '%s'", tag, total, metadata.title)

    return NFTResponse(
        image_hash=image_hash,
        metadata=metadata,
        duration_ms=int(total * 1000),
    )


async def process_image(raw_bytes: bytes) -> NFTResponse:
    image_hash = compute_sha256(raw_bytes)
    tag = image_hash[:12]
    logger.info("[%s] Received image (%d bytes)", tag, len(raw_bytes))

    cached = cache.get(image_hash)
    if cached is not None:
        logger.info("[%s] Cache hit", tag)
        return cached.model_copy(update={"cached": True})

    logger.info("[%s] Cache miss — starting inference (timeout=%ds)", tag, INFERENCE_TIMEOUT)

    loop = asyncio.get_running_loop()
    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                partial(_blocking_inference, raw_bytes, image_hash),
            ),
            timeout=INFERENCE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("[%s] Inference timed out after %ds", tag, INFERENCE_TIMEOUT)
        raise HTTPException(504, f"Model inference timed out after {INFERENCE_TIMEOUT}s. The server CPU may be too slow.")
    except HTTPException:
        raise
    except Exception:
        logger.exception("[%s] Inference failed", tag)
        raise HTTPException(500, "Model inference failed. Check server logs.")

    cache.set(image_hash, response)
    return response
