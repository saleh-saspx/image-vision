from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import HealthResponse, NFTResponse
from app.services.pipeline import process_image
from app.services.vision import vision_service

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_TYPES = {"image/jpeg", "image/png"}


@router.post("/generate", response_model=NFTResponse)
async def generate_nft_metadata(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use JPEG or PNG.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size is {MAX_FILE_SIZE // (1024*1024)}MB.")

    if len(raw_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded.")

    return await process_image(raw_bytes)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", model_loaded=vision_service.is_loaded)
