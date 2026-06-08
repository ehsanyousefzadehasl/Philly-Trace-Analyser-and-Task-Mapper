from __future__ import annotations

import random
from collections.abc import Iterable

import pandas as pd

from collections.abc import Iterable, Mapping

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
    minimum_jobs_by_gpu_count: Mapping[int, int] | None = None,
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

    minimum_counts = {
        int(gpu_count): int(minimum)
        for gpu_count, minimum in (
            minimum_jobs_by_gpu_count or {}
        ).items()
        if int(minimum) > 0
    }

    for gpu_count, minimum in minimum_counts.items():
        if gpu_count not in supported_counts:
            raise ValueError(
                "minimum_jobs_by_gpu_count contains an "
                f"unsupported GPU count: {gpu_count}"
            )

        if minimum < 0:
            raise ValueError(
                "minimum_jobs_by_gpu_count values must be "
                "nonnegative"
            )

    if sum(minimum_counts.values()) > num_jobs:
        raise ValueError(
            "The sum of minimum_jobs_by_gpu_count exceeds "
            "num_jobs"
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
    valid_start_indices: list[int] = []

    for start_index in range(possible_start_count):
        candidate = eligible.iloc[
            start_index:start_index + num_jobs
        ]

        candidate_counts = (
            candidate["source_num_gpus"]
            .value_counts()
            .to_dict()
        )

        satisfies_constraints = all(
            int(candidate_counts.get(gpu_count, 0))
            >= minimum
            for gpu_count, minimum in minimum_counts.items()
        )

        if satisfies_constraints:
            valid_start_indices.append(start_index)

    if not valid_start_indices:
        raise ValueError(
            "No contiguous eligible window satisfies "
            f"minimum_jobs_by_gpu_count={minimum_counts}"
        )

    start_index = random.Random(seed).choice(
        valid_start_indices
    )
    end_index = start_index + num_jobs

    window = eligible.iloc[start_index:end_index].copy()

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

    selected_gpu_counts = {
        str(int(gpu_count)): int(count)
        for gpu_count, count in (
            window["source_num_gpus"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    selection_method = (
        "random_contiguous_eligible_jobs_with_constraints"
        if minimum_counts
        else "random_contiguous_eligible_jobs"
    )

    metadata = {
    "selection_method": selection_method,
        "seed": int(seed),
        "requested_job_count": int(num_jobs),
        "source_job_count": int(len(jobs)),
        "eligible_job_count": int(len(eligible)),
        "filtered_job_count": int(len(jobs) - len(eligible)),
        "supported_gpu_counts": list(supported_counts),
        "minimum_jobs_by_gpu_count": {
            str(key): int(value)
            for key, value in sorted(minimum_counts.items())
        },
        "selected_jobs_by_gpu_count": selected_gpu_counts,
        "candidate_window_count": int(
            len(valid_start_indices)
        ),
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