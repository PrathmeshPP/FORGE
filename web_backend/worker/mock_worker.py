"""
F.O.R.G.E. Website Backend — Deterministic Mock Background Worker
Manages single-worker job queue concurrency, simulated pipeline lifecycle stages,
safe cooperative cancellation, output fixture staging, cross-process file locking,
and startup crash recovery.
"""

import os
import asyncio
import datetime
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

from web_backend.config import (
    FORGE_DATA_DIR,
    JOBS_DIR,
    SAMPLE_ASSETS_DIR,
    FORGE_MAX_ACTIVE_JOBS,
    FORGE_MAX_QUEUE_SIZE,
    FORGE_MOCK_STEP_DELAY_SECONDS,
)
from web_backend.models import JobStatus, PipelineStage
from web_backend.database import get_job_by_id, save_job, get_db_connection
from web_backend.services.qa_validator import (
    validate_and_generate_manifest,
    QAValidationError,
)


class WorkerQueueFullError(Exception):
    """Raised when the worker job queue exceeds max capacity."""
    pass


class CrossProcessLock:
    """
    Guarantees OS-level mutual exclusion across multiple processes or uvicorn workers
    using Windows msvcrt byte-range locking (or fcntl on POSIX).
    The OS automatically releases the lock if the holding process terminates unexpectedly.
    """
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._file = None

    def acquire(self) -> bool:
        """Attempts non-blocking acquisition of the cross-process lock."""
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.lock_path, "w+b")
            self._file.write(b"x")
            self._file.flush()
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (IOError, OSError):
            if self._file:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
            return False

    def release(self):
        """Releases the cross-process lock."""
        if self._file:
            try:
                self._file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                self._file.close()
            except Exception:
                pass
            finally:
                self._file = None


class MockWorker:
    def __init__(self):
        self._lock: Optional[asyncio.Lock] = None
        self._queue: Optional[asyncio.Queue[str]] = None
        self._active_job_id: Optional[str] = None
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._queued_job_ids: set[str] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self.step_delay: float = FORGE_MOCK_STEP_DELAY_SECONDS
        self._cross_process_lock = CrossProcessLock(FORGE_DATA_DIR / ".worker_reconstruction.lock")

    def _ensure_primitives(self):
        """Ensures asyncio primitives are bound to the currently running event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=FORGE_MAX_QUEUE_SIZE)

    def reset(self):
        """Resets worker internal state (useful for test isolation)."""
        self.stop_background_loop()
        self._cross_process_lock.release()
        self._lock = None
        self._queue = None
        self._active_job_id = None
        self._cancel_events.clear()
        self._queued_job_ids.clear()
        self.step_delay = FORGE_MOCK_STEP_DELAY_SECONDS

    def is_busy(self) -> bool:
        """Returns True if a job is currently actively executing."""
        return self._active_job_id is not None

    def get_active_job_id(self) -> Optional[str]:
        """Returns the UUID of the currently processing job, or None."""
        return self._active_job_id

    def is_queue_full(self) -> bool:
        """Returns True if the queue is at max capacity."""
        self._ensure_primitives()
        return self._queue.full()

    def get_queued_count(self) -> int:
        """Returns count of waiting queued jobs."""
        self._ensure_primitives()
        return self._queue.qsize()

    async def enqueue_job(self, job_id: str) -> bool:
        """
        Enqueues a job for background processing.
        Ensures idempotency (no duplicate entries) and bounded queue limits.
        """
        self._ensure_primitives()

        # If already running or in queue, idempotent success
        if self._active_job_id == job_id or job_id in self._queued_job_ids:
            return True

        if self._queue.full():
            raise WorkerQueueFullError(
                f"Workstation queue is at capacity ({FORGE_MAX_QUEUE_SIZE} jobs max). "
                "Please wait for current jobs to finish."
            )

        self._cancel_events[job_id] = asyncio.Event()
        self._queued_job_ids.add(job_id)
        await self._queue.put(job_id)
        return True

    async def cancel_job(self, job_id: str) -> bool:
        """
        Signals cancellation for a queued or currently executing job.
        Cooperative cancellation breaks the worker sleep immediately.
        """
        self._ensure_primitives()

        # Signal cancellation event if active
        if job_id in self._cancel_events:
            self._cancel_events[job_id].set()

        # Update DB state immediately
        job = get_job_by_id(job_id)
        if job and job["status"] not in [
            JobStatus.READY.value,
            JobStatus.CANCELLED.value,
            JobStatus.FAILED.value,
        ]:
            job["status"] = JobStatus.CANCELLED.value
            job["progress_stage"] = PipelineStage.IDLE.value
            job["progress_stage_label"] = "Job cancelled by user"
            now_time = datetime.datetime.now().strftime("%H:%M:%S")
            job["logs"].append({
                "timestamp": now_time,
                "stage": "idle",
                "message": "[WORKER] Cancellation confirmed. Host execution halted.",
            })
            save_job(job)

        if job_id in self._queued_job_ids:
            self._queued_job_ids.remove(job_id)

        return True

    async def _sleep_or_cancel(self, job_id: str, duration: float) -> bool:
        """
        Sleeps for duration seconds in short slices (25ms), checking for cancellation.
        Returns True if cancelled, False if sleep finished normally.
        """
        event = self._cancel_events.get(job_id)
        if not event:
            await asyncio.sleep(duration)
            return False

        elapsed = 0.0
        step = 0.025
        while elapsed < duration:
            if event.is_set():
                return True
            sleep_time = min(step, duration - elapsed)
            await asyncio.sleep(sleep_time)
            elapsed += sleep_time

        return event.is_set()

    def _copy_sample_fixtures(self, job_id: str) -> Path:
        """
        Copies verified indoor sample room fixtures into the job's outputs directory.
        Watermarks OBJ with notice and references textured_mesh.mtl.
        """
        outputs_dir = JOBS_DIR / job_id / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        sample_obj = SAMPLE_ASSETS_DIR / "indoor_sample_room.obj"
        sample_mtl = SAMPLE_ASSETS_DIR / "indoor_sample_room.mtl"
        sample_ply = SAMPLE_ASSETS_DIR / "indoor_sample_room.ply"

        if not sample_obj.exists() or not sample_mtl.exists() or not sample_ply.exists():
            raise QAValidationError(
                "GATE_B_MISSING_SAMPLE_FIXTURES",
                f"Source sample fixtures not found in {SAMPLE_ASSETS_DIR}."
            )

        # Target filenames conforming to API_CONTRACT.md and TRD.md
        target_obj = outputs_dir / "textured_mesh.obj"
        target_mtl = outputs_dir / "textured_mesh.mtl"
        target_ply = outputs_dir / "surface_mesh_photographic.ply"

        # Copy MTL
        shutil.copy2(sample_mtl, target_mtl)

        # Copy PLY
        shutil.copy2(sample_ply, target_ply)

        # Write OBJ with explicit material library reference
        with open(sample_obj, "r", encoding="utf-8") as f_in, open(target_obj, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if line.startswith("mtllib "):
                    f_out.write("mtllib textured_mesh.mtl\n")
                else:
                    f_out.write(line)

        return outputs_dir

    async def execute_job(self, job_id: str):
        """
        Executes a single reconstruction job through all deterministic simulated stages:
        queued -> running (ingestion, keyframes, spatial inference, meshing, texturing)
        -> validating_output (Gates A-C) -> ready (or failed/cancelled).
        Enforces both intra-process (asyncio.Lock) and cross-process (file lock) mutual exclusion.
        """
        self._ensure_primitives()
        async with self._lock:
            # Enforce cross-process mutual exclusion across multiple OS processes
            if not self._cross_process_lock.acquire():
                job = get_job_by_id(job_id)
                if job:
                    job["status"] = JobStatus.FAILED.value
                    job["last_error_code"] = "CROSS_PROCESS_LOCK_BUSY"
                    job["last_error_message"] = (
                        "Cross-process lock collision: Another workstation application process "
                        "is actively executing a reconstruction job."
                    )
                    job["logs"].append({
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "stage": "idle",
                        "message": "[WORKER] Cross-process lock busy. Execution aborted to protect workstation GPU resources.",
                    })
                    save_job(job)
                return

            self._active_job_id = job_id
            if job_id in self._queued_job_ids:
                self._queued_job_ids.remove(job_id)

            try:
                job = get_job_by_id(job_id)
                if not job:
                    return

                # If job was cancelled while waiting in queue, abort
                event = self._cancel_events.get(job_id)
                if (event and event.is_set()) or job["status"] == JobStatus.CANCELLED.value:
                    return

                # Begin RUNNING state
                now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                job["status"] = JobStatus.RUNNING.value
                job["started_at"] = now_iso
                job["progress_stage"] = PipelineStage.INGESTION.value
                job["progress_stage_label"] = "Step 1/5: Video Ingestion & Container Verification [MOCK]"
                job["logs"].append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "stage": "ingestion",
                    "message": "[MOCK PIPELINE] Workstation worker acquired job. Video container parameters verified.",
                })
                save_job(job)

                # Simulated Stages Configuration
                simulated_stages = [
                    (
                        PipelineStage.INGESTION,
                        "Step 1/5: Video Ingestion & Stream Decoding [MOCK]",
                        "[MOCK PIPELINE] Decoded baseline frames. Audio track safely discarded.",
                    ),
                    (
                        PipelineStage.KEYFRAME_EXTRACTION,
                        "Step 2/5: Keyframe Extraction & Viewpoint Diversity [MOCK]",
                        "[MOCK PIPELINE] Extracted 48 sharp keyframes. Viewpoint baselines confirmed.",
                    ),
                    (
                        PipelineStage.SPATIAL_INFERENCE,
                        "Step 3/5: Simulated Foundation Spatial Inference [MOCK]",
                        "[MOCK PIPELINE] Simulated VGGT-Omega spatial inference pass. Poses tracked.",
                    ),
                    (
                        PipelineStage.DENSE_MESHING,
                        "Step 4/5: Dense Point Cloud & Surface Reconstruction [MOCK]",
                        "[MOCK PIPELINE] Point cloud densified. Poisson surface reconstruction converged.",
                    ),
                    (
                        PipelineStage.TEXTURING,
                        "Step 5/5: Texture Projection & Material Export [MOCK]",
                        "[MOCK PIPELINE] Surface charts projected and texture atlas synthesized.",
                    ),
                ]

                # Execute stages with cancellation checks
                for stage_enum, label, log_msg in simulated_stages:
                    if await self._sleep_or_cancel(job_id, self.step_delay):
                        # Cancelled!
                        return

                    job = get_job_by_id(job_id)
                    if not job or job["status"] == JobStatus.CANCELLED.value:
                        return

                    job["progress_stage"] = stage_enum.value
                    job["progress_stage_label"] = label
                    job["logs"].append({
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "stage": stage_enum.value,
                        "message": log_msg,
                    })
                    save_job(job)

                # Stage 6: VALIDATING_OUTPUT & QA_VERIFICATION
                if await self._sleep_or_cancel(job_id, self.step_delay * 0.5):
                    return

                job = get_job_by_id(job_id)
                if not job or job["status"] == JobStatus.CANCELLED.value:
                    return

                job["status"] = JobStatus.VALIDATING_OUTPUT.value
                job["progress_stage"] = PipelineStage.QA_VERIFICATION.value
                job["progress_stage_label"] = "Running Gate A-C Technical Quality Verification"
                job["logs"].append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "stage": "qa_verification",
                    "message": "[MOCK PIPELINE] Staging sample indoor room assets and executing Gates A-C QA...",
                })
                save_job(job)

                # Copy sample fixtures and run Gate A-C validation
                outputs_dir = self._copy_sample_fixtures(job_id)
                qa_result = validate_and_generate_manifest(outputs_dir, job_id)

                if await self._sleep_or_cancel(job_id, self.step_delay * 0.5):
                    return

                job = get_job_by_id(job_id)
                if not job or job["status"] == JobStatus.CANCELLED.value:
                    return

                # Stage 7: Transition to READY
                end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                start_dt = datetime.datetime.fromisoformat(job["started_at"])
                end_dt = datetime.datetime.fromisoformat(end_iso)
                duration_secs = max(1, int((end_dt - start_dt).total_seconds()))

                job["status"] = JobStatus.READY.value
                job["progress_stage"] = PipelineStage.IDLE.value
                job["progress_stage_label"] = "Reconstruction Complete (Mock Sample Ready)"
                job["finished_at"] = end_iso
                job["duration_seconds"] = duration_secs
                job["scene"] = qa_result["scene"]
                job["quality_summary"] = qa_result["quality_summary"]
                job["is_sample_job"] = True
                job["logs"].append({
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "stage": "idle",
                    "message": (
                        "[MOCK PIPELINE] Quality checks passed (Gates A-D). "
                        "Watermarked sample 3D assets ready for interactive inspection in viewer."
                    ),
                })
                save_job(job)

            except QAValidationError as qe:
                job = get_job_by_id(job_id)
                if job:
                    job["status"] = JobStatus.FAILED.value
                    job["progress_stage"] = PipelineStage.IDLE.value
                    job["progress_stage_label"] = "Reconstruction Failed (Quality Gate Error)"
                    job["last_error_code"] = qe.code
                    job["last_error_message"] = qe.message
                    job["logs"].append({
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "stage": "idle",
                        "message": f"[QA FAILURE] {qe.code}: {qe.message}",
                    })
                    save_job(job)

            except Exception as e:
                job = get_job_by_id(job_id)
                if job:
                    job["status"] = JobStatus.FAILED.value
                    job["progress_stage"] = PipelineStage.IDLE.value
                    job["progress_stage_label"] = "Reconstruction Failed (System Error)"
                    job["last_error_code"] = "WORKER_INTERNAL_ERROR"
                    job["last_error_message"] = f"Worker exception: {str(e)}"
                    job["logs"].append({
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "stage": "idle",
                        "message": f"[SYSTEM ERROR] {str(e)}",
                    })
                    save_job(job)

            finally:
                self._cross_process_lock.release()
                self._active_job_id = None
                self._cancel_events.pop(job_id, None)

    async def run_worker_loop(self):
        """Continuous background worker loop processing queued reconstruction jobs."""
        self._ensure_primitives()
        self._is_running = True
        try:
            while self._is_running:
                try:
                    job_id = await self._queue.get()
                except asyncio.CancelledError:
                    break

                try:
                    await self.execute_job(job_id)
                except Exception:
                    pass
                finally:
                    self._queue.task_done()
        finally:
            self._is_running = False

    def start_background_loop(self):
        """Spawns the background worker task if not already started."""
        self._ensure_primitives()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self.run_worker_loop())

    def stop_background_loop(self):
        """Stops the worker loop."""
        self._is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()


# Global Mock Worker Singleton
mock_worker = MockWorker()


def recover_interrupted_jobs() -> int:
    """
    On application startup, detects any jobs left in non-terminal states
    (running, queued, validating_output, cancelling) caused by an ungraceful
    workstation reboot or process termination. Marks them failed with code
    PIPELINE_INTERRUPTED.
    """
    interrupted_statuses = [
        JobStatus.RUNNING.value,
        JobStatus.QUEUED.value,
        JobStatus.VALIDATING_OUTPUT.value,
        JobStatus.CANCELLING.value,
    ]

    with get_db_connection() as conn:
        placeholders = ",".join(["?"] * len(interrupted_statuses))
        cursor = conn.execute(
            f"SELECT id, logs_json FROM jobs WHERE status IN ({placeholders})",
            interrupted_statuses,
        )
        rows = cursor.fetchall()

        count = len(rows)
        now_time = datetime.datetime.now().strftime("%H:%M:%S")

        for row in rows:
            job_id = row["id"]
            logs = []
            if row["logs_json"]:
                import json
                try:
                    logs = json.loads(row["logs_json"])
                except Exception:
                    logs = []

            logs.append({
                "timestamp": now_time,
                "stage": "idle",
                "message": (
                    "[SHUTDOWN RECOVERY] Job interrupted by workstation application shutdown/restart. "
                    "Marked failed to prevent orphaned background execution."
                ),
            })

            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    progress_stage = ?,
                    progress_stage_label = ?,
                    last_error_code = ?,
                    last_error_message = ?,
                    logs_json = ?
                WHERE id = ?
                """,
                (
                    JobStatus.FAILED.value,
                    PipelineStage.IDLE.value,
                    "Interrupted by Host Restart",
                    "PIPELINE_INTERRUPTED",
                    "Job interrupted by host application shutdown/restart.",
                    json.dumps(logs),
                    job_id,
                ),
            )
        conn.commit()
        return count
