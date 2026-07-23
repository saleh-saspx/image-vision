#!/usr/bin/env python3
"""Startup script for the NFT Metadata Generator service."""
import logging

import uvicorn

from app.core.config import settings

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if settings.WORKERS > 1:
        # The job store and queue live in process memory, so a client polling
        # /jobs/{id} would hit a process that never saw its job. Scale with
        # INFERENCE_WORKERS (threads), or move the store to Redis first.
        logger.warning(
            "WORKERS=%d ignored: the inference queue is in-process. "
            "Running a single uvicorn worker; set INFERENCE_WORKERS to add threads.",
            settings.WORKERS,
        )

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,
        log_level=settings.LOG_LEVEL,
        timeout_keep_alive=180,
    )
