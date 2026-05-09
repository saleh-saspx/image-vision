import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import HealthResponse, NFTResponse
from app.services.pipeline import process_image
from app.services.vision import vision_service

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {"image/jpeg", "image/png"}


@router.post("/generate", response_model=NFTResponse)
async def generate_nft_metadata(file: UploadFile = File(...)):
    logger.info("Received upload: name=%s type=%s", file.filename, file.content_type)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use JPEG or PNG.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size is {MAX_FILE_SIZE // (1024*1024)}MB.")

    if len(raw_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")

    t0 = time.monotonic()
    result = await process_image(raw_bytes)
    logger.info("Request completed in %.2fs", time.monotonic() - t0)
    return result


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", model_loaded=vision_service.is_loaded)


@router.get("/debug")
async def debug_info():
    """Quick diagnostics — no image needed."""
    import torch
    from app.core.cache import cache

    return {
        "model_loaded": vision_service.is_loaded,
        "device": str(vision_service._device) if vision_service._device else "not set",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cache_size": cache.size,
    }
