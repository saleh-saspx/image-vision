import asyncio
import logging
import time
from functools import partial

from fastapi import HTTPException

from app.core.cache import cache
from app.models.schemas import NFTResponse
from app.services.nft_generator import generate_metadata
from app.services.vision import vision_service
from app.utils.hash import compute_sha256
from app.utils.image import preprocess_image

logger = logging.getLogger(__name__)

INFERENCE_TIMEOUT = 120  # seconds


def _blocking_inference(raw_bytes: bytes, image_hash: str) -> NFTResponse:
    """CPU-bound work that must NOT run on the event loop."""
    t0 = time.monotonic()

    logger.info("[%s] Preprocessing image...", image_hash[:12])
    image = preprocess_image(raw_bytes)
    t1 = time.monotonic()
    logger.info("[%s] Preprocessing done (%.2fs). Running model inference...", image_hash[:12], t1 - t0)

    attributes = vision_service.analyze(image)
    t2 = time.monotonic()
    logger.info("[%s] Inference done (%.2fs). Attributes: %s", image_hash[:12], t2 - t1, attributes)

    metadata = generate_metadata(attributes, image_hash)
    logger.info("[%s] Metadata generated. Total: %.2fs", image_hash[:12], time.monotonic() - t0)

    return NFTResponse(image_hash=image_hash, metadata=metadata)


async def process_image(raw_bytes: bytes) -> NFTResponse:
    image_hash = compute_sha256(raw_bytes)
    logger.info("[%s] Received image (%d bytes)", image_hash[:12], len(raw_bytes))

    cached = cache.get(image_hash)
    if cached is not None:
        logger.info("[%s] Cache hit — returning cached result", image_hash[:12])
        return cached

    logger.info("[%s] Cache miss — starting inference (timeout=%ds)", image_hash[:12], INFERENCE_TIMEOUT)

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
        logger.error("[%s] Inference timed out after %ds", image_hash[:12], INFERENCE_TIMEOUT)
        raise HTTPException(504, f"Model inference timed out after {INFERENCE_TIMEOUT}s. The server CPU may be too slow.")
    except Exception:
        logger.exception("[%s] Inference failed", image_hash[:12])
        raise HTTPException(500, "Model inference failed. Check server logs.")

    cache.set(image_hash, response)
    return response
