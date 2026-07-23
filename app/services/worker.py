"""Background inference queue.

Uploads return immediately with a job id; the actual model run happens here, on
dedicated threads, so a slow CPU produces a longer wait rather than a gateway
timeout. Clients poll GET /jobs/{id} instead of holding a connection open.

Worker count defaults to 1 on purpose: VisionService serialises on its own lock
and torch already uses every core, so extra workers would add memory pressure
and queue-jumping without adding throughput.
"""
from __future__ import annotations

import logging
import os
import queue
import threading

from app.core.cache import cache
from app.core.jobs import JobStore, job_store
from app.services.pipeline import run_inference

logger = logging.getLogger(__name__)

# Bounded on purpose: each waiting item pins its raw upload bytes in memory, so
# an unbounded queue turns a traffic spike into an OOM. Full queue -> 503.
DEFAULT_MAX_QUEUE = int(os.getenv("QUEUE_MAX_SIZE", "16"))
DEFAULT_WORKERS = int(os.getenv("INFERENCE_WORKERS", "1"))

_SHUTDOWN = object()


class InferenceWorker:
    def __init__(
        self,
        store: JobStore | None = None,
        max_queue: int = DEFAULT_MAX_QUEUE,
        workers: int = DEFAULT_WORKERS,
    ):
        self._store = store if store is not None else job_store
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._threads: list[threading.Thread] = []
        self._worker_count = max(1, workers)
        self._max_queue = max_queue
        self._running = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for index in range(self._worker_count):
            thread = threading.Thread(
                target=self._run,
                name=f"inference-worker-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        logger.info("Inference queue started (workers=%d, capacity=%d)", self._worker_count, self._max_queue)

    def stop(self, timeout: float = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        for _ in self._threads:
            self._queue.put(_SHUTDOWN)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()
        logger.info("Inference queue stopped")

    # -- producer -----------------------------------------------------------

    def submit(self, job_id: str, raw_bytes: bytes) -> bool:
        """Enqueue work. Returns False if the queue is full (caller should 503)."""
        try:
            self._queue.put_nowait((job_id, raw_bytes))
            return True
        except queue.Full:
            logger.warning("[%s] Queue full (capacity=%d) — rejecting", job_id[:12], self._max_queue)
            return False

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        return self._max_queue

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def is_running(self) -> bool:
        return self._running

    # -- consumer -----------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SHUTDOWN:
                    return
                self._process(*item)
            finally:
                self._queue.task_done()

    def _process(self, job_id: str, raw_bytes: bytes) -> None:
        tag = job_id[:12]
        self._store.mark_processing(job_id)
        try:
            response = run_inference(raw_bytes, job_id)
            cache.set(job_id, response)
            self._store.mark_completed(job_id, response)
            logger.info("[%s] Job completed", tag)
        except Exception as exc:
            # A failed job must never take the worker thread down with it —
            # the next queued upload still has to run.
            logger.exception("[%s] Job failed", tag)
            self._store.mark_failed(job_id, f"{type(exc).__name__}: {exc}")


inference_worker = InferenceWorker()
