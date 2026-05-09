import asyncio
from functools import partial

from app.core.cache import cache
from app.models.schemas import NFTResponse
from app.services.nft_generator import generate_metadata
from app.services.vision import vision_service
from app.utils.hash import compute_sha256
from app.utils.image import preprocess_image


def _blocking_inference(raw_bytes: bytes, image_hash: str) -> NFTResponse:
    """CPU-bound work that must NOT run on the event loop."""
    image = preprocess_image(raw_bytes)
    attributes = vision_service.analyze(image)
    metadata = generate_metadata(attributes, image_hash)
    return NFTResponse(image_hash=image_hash, metadata=metadata)


async def process_image(raw_bytes: bytes) -> NFTResponse:
    image_hash = compute_sha256(raw_bytes)

    cached = cache.get(image_hash)
    if cached is not None:
        return cached

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        partial(_blocking_inference, raw_bytes, image_hash),
    )

    cache.set(image_hash, response)
    return response
