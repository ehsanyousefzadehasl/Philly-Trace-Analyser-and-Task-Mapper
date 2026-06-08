from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PHILLY_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_philly_time(value) -> datetime | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None

    try:
        return datetime.strptime(text, PHILLY_TIME_FORMAT)
    except ValueError:
        return None


def count_attempt_gpus(attempt: dict) -> int:
    gpu_entries = {
        (str(detail.get("ip", "")), str(gpu))
        for detail in (attempt.get("detail") or [])
        for gpu in (detail.get("gpus") or [])
    }
    return len(gpu_entries)


def load_philly_jobs(path: Path) -> pd.DataFrame:
    jobs = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(jobs, list):
        raise ValueError("Philly trace root must contain a list of jobs")

    rows: list[dict] = []

    for job in jobs:
        if job.get("status") != "Pass":
            continue

        attempts = job.get("attempts") or []
        if not attempts:
            continue

        successful_attempt = attempts[-1]

        submitted_at = parse_philly_time(job.get("submitted_time"))
        started_at = parse_philly_time(
            successful_attempt.get("start_time")
        )
        ended_at = parse_philly_time(
            successful_attempt.get("end_time")
        )

        if (
            submitted_at is None
            or started_at is None
            or ended_at is None
        ):
            continue

        duration_s = (ended_at - started_at).total_seconds()
        queue_wait_s = (started_at - submitted_at).total_seconds()
        num_gpus = count_attempt_gpus(successful_attempt)

        if duration_s <= 0 or queue_wait_s < 0 or num_gpus <= 0:
            continue

        details = successful_attempt.get("detail") or []

        rows.append(
            {
                "source_trace": "philly",
                "source_cluster": "philly",
                "source_job_id": str(job.get("jobid")),
                "source_status": str(job.get("status")),
                "submitted_at": submitted_at,
                "source_start_at": started_at,
                "source_end_at": ended_at,
                "source_duration_s": float(duration_s),
                "source_queue_wait_s": float(queue_wait_s),
                "source_num_gpus": int(num_gpus),
                "source_attempt_count": len(attempts),
                "source_multi_node": len(details) > 1,
            }
        )

    if not rows:
        raise ValueError("No valid passed jobs found in Philly trace")

    result = pd.DataFrame(rows).sort_values(
        ["submitted_at", "source_job_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    first_submission = result["submitted_at"].iloc[0]

    result["submit_time_s"] = (
        result["submitted_at"] - first_submission
    ).dt.total_seconds()

    result["source_interarrival_s"] = (
        result["submitted_at"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )

    return result