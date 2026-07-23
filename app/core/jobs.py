"""In-memory job registry for queued inference.

The image's SHA-256 doubles as the job id. That is deliberate: uploading the
same image twice joins the existing job instead of queueing a duplicate, so a
client retrying after a dropped connection never pays for inference twice.

State lives in this process only. With more than one uvicorn worker, a client
polling could land on a process that never saw the job — run a single worker
(the default) or move this behind Redis before scaling out.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from app.models.schemas import JobStatus, NFTResponse

# Finished jobs stay readable for a while so a slow client can still collect
# its result, then are purged to bound memory.
DEFAULT_RESULT_TTL = 3600  # seconds
DEFAULT_MAX_JOBS = 2048


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: NFTResponse | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED)

    @property
    def duration_ms(self) -> int | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return int((self.finished_at - self.started_at) * 1000)

    @property
    def waited_ms(self) -> int:
        started = self.started_at if self.started_at is not None else time.time()
        return int((started - self.created_at) * 1000)


class JobStore:
    def __init__(self, result_ttl: int = DEFAULT_RESULT_TTL, max_jobs: int = DEFAULT_MAX_JOBS):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._result_ttl = result_ttl
        self._max_jobs = max_jobs

    def create_or_get(self, job_id: str) -> tuple[Job, bool]:
        """Return (job, created). `created` is False when joining an existing job."""
        with self._lock:
            self._purge_locked()

            existing = self._jobs.get(job_id)
            if existing is not None:
                return existing, False

            job = Job(id=job_id)
            self._jobs[job_id] = job
            return job, True

    def complete_now(self, job_id: str, result: NFTResponse) -> Job:
        """Register an already-finished job — used for cache hits, which never
        touch the queue."""
        now = time.time()
        with self._lock:
            job = Job(
                id=job_id,
                status=JobStatus.COMPLETED,
                created_at=now,
                started_at=now,
                finished_at=now,
                result=result,
            )
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_processing(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = JobStatus.PROCESSING
                job.started_at = time.time()

    def mark_completed(self, job_id: str, result: NFTResponse) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = JobStatus.COMPLETED
                job.result = result
                job.finished_at = time.time()

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = JobStatus.FAILED
                job.error = error
                job.finished_at = time.time()

    def discard(self, job_id: str) -> None:
        """Drop a job that was never actually queued (e.g. rejected backpressure)."""
        with self._lock:
            self._jobs.pop(job_id, None)

    def queue_position(self, job_id: str) -> int | None:
        """1-based position among still-waiting jobs; None once it starts."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != JobStatus.QUEUED:
                return None
            ahead = sum(
                1
                for other in self._jobs.values()
                if other.status == JobStatus.QUEUED and other.created_at < job.created_at
            )
            return ahead + 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._purge_locked()
            counts = {status.value: 0 for status in JobStatus}
            for job in self._jobs.values():
                counts[job.status.value] += 1
            counts["total"] = len(self._jobs)
            return counts

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    # -- internal -----------------------------------------------------------

    def _purge_locked(self) -> None:
        """Evict expired terminal jobs. Never evicts queued or running work."""
        cutoff = time.time() - self._result_ttl
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.is_terminal and (job.finished_at or job.created_at) < cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]

        # Hard cap as a backstop: drop the oldest finished jobs first.
        if len(self._jobs) > self._max_jobs:
            terminal = sorted(
                (job for job in self._jobs.values() if job.is_terminal),
                key=lambda j: j.finished_at or j.created_at,
            )
            for job in terminal[: len(self._jobs) - self._max_jobs]:
                del self._jobs[job.id]


job_store = JobStore()
