"""In-memory хранилище фоновых задач."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Job:
    job_id: str
    status: Literal["pending", "running", "done", "error"] = "pending"
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


_store: dict[str, Job] = {}
_lock = asyncio.Lock()


def new_job() -> Job:
    job = Job(job_id=str(uuid.uuid4()))
    _store[job.job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _store.get(job_id)


async def update_job(
    job_id: str,
    status: Literal["pending", "running", "done", "error"],
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    async with _lock:
        job = _store.get(job_id)
        if job is None:
            return
        job.status = status
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
