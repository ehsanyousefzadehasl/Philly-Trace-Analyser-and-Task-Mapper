from __future__ import annotations

import random
from collections.abc import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "source_job_id",
    "source_num_gpus",
    "submit_time_s",
    "source_interarrival_s",
}


def select_contiguous_job_window(
    source_jobs: pd.DataFrame,
    *,
    num_jobs: int,
    supported_gpu_counts: Iterable[int] = (1, 2),
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED_COLUMNS - set(source_jobs.columns)
    if missing:
        raise ValueError(
            "source_jobs is missing required columns: "
            f"{sorted(missing)}"
        )

    if num_jobs <= 0:
        raise ValueError("num_jobs must be positive")

    supported_counts = tuple(
        sorted({int(value) for value in supported_gpu_counts})
    )

    if not supported_counts:
        raise ValueError(
            "supported_gpu_counts must not be empty"
        )

    if any(value <= 0 for value in supported_counts):
        raise ValueError(
            "supported_gpu_counts must contain positive integers"
        )

    jobs = source_jobs.copy()

    jobs["source_num_gpus"] = pd.to_numeric(
        jobs["source_num_gpus"],
        errors="raise",
    ).astype(int)

    jobs["submit_time_s"] = pd.to_numeric(
        jobs["submit_time_s"],
        errors="raise",
    )

    jobs = jobs.sort_values(
        ["submit_time_s", "source_job_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    eligible = jobs.loc[
        jobs["source_num_gpus"].isin(supported_counts)
    ].copy()

    if len(eligible) < num_jobs:
        raise ValueError(
            f"Requested {num_jobs} jobs, but only "
            f"{len(eligible)} eligible jobs are available"
        )

    possible_start_count = len(eligible) - num_jobs + 1
    start_index = random.Random(seed).randrange(
        possible_start_count
    )
    end_index = start_index + num_jobs

    window = eligible.iloc[start_index:end_index].copy()

    # Preserve positions and timing from the complete normalized trace.
    window["source_trace_submit_time_s"] = window[
        "submit_time_s"
    ]
    window["source_trace_interarrival_s"] = window[
        "source_interarrival_s"
    ]

    first_submit_time = float(window["submit_time_s"].iloc[0])

    window["submit_time_s"] = (
        window["submit_time_s"] - first_submit_time
    )

    window["source_interarrival_s"] = (
        window["submit_time_s"]
        .diff()
        .fillna(0.0)
    )

    window["source_window_job_index"] = range(len(window))

    window = window.reset_index(drop=True)

    metadata = {
        "selection_method": "random_contiguous_eligible_jobs",
        "seed": int(seed),
        "requested_job_count": int(num_jobs),
        "source_job_count": int(len(jobs)),
        "eligible_job_count": int(len(eligible)),
        "filtered_job_count": int(len(jobs) - len(eligible)),
        "supported_gpu_counts": list(supported_counts),
        "eligible_start_index": int(start_index),
        "eligible_end_index_exclusive": int(end_index),
        "first_source_job_id": str(
            window["source_job_id"].iloc[0]
        ),
        "last_source_job_id": str(
            window["source_job_id"].iloc[-1]
        ),
        "original_first_submit_time_s": first_submit_time,
        "original_last_submit_time_s": float(
            window["source_trace_submit_time_s"].iloc[-1]
        ),
        "selected_arrival_span_s": float(
            window["submit_time_s"].iloc[-1]
        ),
    }

    return window, metadata
