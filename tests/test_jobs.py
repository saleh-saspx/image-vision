import time

from app.core.jobs import Job, JobStore
from app.models.schemas import Attribute, JobStatus, NFTMetadata, NFTResponse


def _response(job_id: str = "abc") -> NFTResponse:
    return NFTResponse(
        image_hash=job_id,
        metadata=NFTMetadata(
            name="Classic Blue Sofa",
            title="Classic Blue Sofa",
            description="A classic blue sofa.",
            attributes=[Attribute(trait_type="Style", value="Classic")],
        ),
    )


class TestLifecycle:
    def test_new_job_starts_queued(self):
        store = JobStore()
        job, created = store.create_or_get("a")
        assert created is True
        assert job.status == JobStatus.QUEUED
        assert job.is_terminal is False

    def test_same_id_joins_existing_job(self):
        store = JobStore()
        first, created_first = store.create_or_get("a")
        second, created_second = store.create_or_get("a")
        assert created_first is True
        assert created_second is False
        assert first is second

    def test_transitions_to_completed(self):
        store = JobStore()
        store.create_or_get("a")
        store.mark_processing("a")
        assert store.get("a").status == JobStatus.PROCESSING

        store.mark_completed("a", _response("a"))
        job = store.get("a")
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert job.is_terminal is True

    def test_failure_records_error_not_result(self):
        store = JobStore()
        store.create_or_get("a")
        store.mark_processing("a")
        store.mark_failed("a", "ValueError: bad json")

        job = store.get("a")
        assert job.status == JobStatus.FAILED
        assert job.result is None
        assert "ValueError" in job.error
        assert job.is_terminal is True

    def test_complete_now_skips_the_queue(self):
        store = JobStore()
        job = store.complete_now("a", _response("a"))
        assert job.status == JobStatus.COMPLETED
        assert store.queue_position("a") is None

    def test_discard_removes_job(self):
        store = JobStore()
        store.create_or_get("a")
        store.discard("a")
        assert store.get("a") is None

    def test_unknown_job_is_none(self):
        assert JobStore().get("nope") is None

    def test_marking_unknown_job_is_a_noop(self):
        store = JobStore()
        store.mark_completed("ghost", _response())
        store.mark_failed("ghost", "boom")
        assert store.get("ghost") is None


class TestQueuePosition:
    def test_positions_follow_arrival_order(self):
        store = JobStore()
        for job_id in ("a", "b", "c"):
            store.create_or_get(job_id)
            time.sleep(0.001)  # distinct created_at
        assert store.queue_position("a") == 1
        assert store.queue_position("b") == 2
        assert store.queue_position("c") == 3

    def test_position_clears_once_running(self):
        store = JobStore()
        store.create_or_get("a")
        store.mark_processing("a")
        assert store.queue_position("a") is None

    def test_running_job_does_not_hold_a_slot(self):
        store = JobStore()
        store.create_or_get("a")
        time.sleep(0.001)
        store.create_or_get("b")
        store.mark_processing("a")
        assert store.queue_position("b") == 1


class TestTimings:
    def test_duration_measured_between_start_and_finish(self):
        store = JobStore()
        store.create_or_get("a")
        store.mark_processing("a")
        time.sleep(0.02)
        store.mark_completed("a", _response("a"))
        assert store.get("a").duration_ms >= 15

    def test_duration_is_none_while_queued(self):
        store = JobStore()
        store.create_or_get("a")
        assert store.get("a").duration_ms is None

    def test_wait_time_is_tracked_separately(self):
        store = JobStore()
        store.create_or_get("a")
        time.sleep(0.02)
        store.mark_processing("a")
        assert store.get("a").waited_ms >= 15


class TestEviction:
    def test_expired_terminal_jobs_are_purged(self):
        store = JobStore(result_ttl=0)
        store.create_or_get("old")
        store.mark_completed("old", _response("old"))
        time.sleep(0.01)  # clock resolution: finished_at must fall before the cutoff
        # Any create triggers a purge sweep.
        store.create_or_get("new")
        assert store.get("old") is None
        assert store.get("new") is not None

    def test_unfinished_jobs_are_never_purged(self):
        store = JobStore(result_ttl=0)
        store.create_or_get("waiting")
        store.create_or_get("other")
        assert store.get("waiting") is not None

    def test_stats_counts_by_status(self):
        store = JobStore()
        store.create_or_get("a")
        store.create_or_get("b")
        store.mark_processing("b")
        store.create_or_get("c")
        store.mark_completed("c", _response("c"))

        stats = store.stats()
        assert stats["queued"] == 1
        assert stats["processing"] == 1
        assert stats["completed"] == 1
        assert stats["total"] == 3


class TestConcurrency:
    def test_create_or_get_yields_one_winner_under_threads(self):
        import threading

        store = JobStore()
        results: list[bool] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            _, created = store.create_or_get("same")
            with lock:
                results.append(created)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one caller may enqueue the work; the rest must join it.
        assert sum(results) == 1


class TestJobModel:
    def test_terminal_states(self):
        assert Job(id="x", status=JobStatus.QUEUED).is_terminal is False
        assert Job(id="x", status=JobStatus.PROCESSING).is_terminal is False
        assert Job(id="x", status=JobStatus.COMPLETED).is_terminal is True
        assert Job(id="x", status=JobStatus.FAILED).is_terminal is True
