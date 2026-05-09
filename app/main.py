import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.vision import vision_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Silence noisy pyvips debug/info logs
logging.getLogger("pyvips").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    vision_service.load_model()
    yield


app = FastAPI(
    title="NFT Metadata Generator",
    description="Generate ERC-721 compatible NFT metadata from images using vision AI.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow configurable origins, default to permissive for development
_allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
