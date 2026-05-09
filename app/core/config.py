import os


class Settings:
    MODEL_ID: str = os.getenv("MODEL_ID", "vikhyatk/moondream2")
    MODEL_REVISION: str = os.getenv("MODEL_REVISION", "2025-01-09")
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "512"))
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))
    IMAGE_TARGET_SIZE: int = int(os.getenv("IMAGE_TARGET_SIZE", "512"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")


settings = Settings()
