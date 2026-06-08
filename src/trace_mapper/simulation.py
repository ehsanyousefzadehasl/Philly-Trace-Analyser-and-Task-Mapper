from __future__ import annotations

import heapq

import pandas as pd


REQUIRED_COLUMNS = {
    "submit_time_s",
    "task_path",
    "mapped_num_gpus",
    "mapped_solo_runtime_s",
}


def simulate_exclusive_execution(
    trace: pd.DataFrame,
    *,
    server_gpu_count: int,
) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED_COLUMNS - set(trace.columns)
    if missing:
        raise ValueError(
            f"Trace is missing required columns: {sorted(missing)}"
        )

    if server_gpu_count <= 0:
        raise ValueError("server_gpu_count must be positive")

    jobs = trace.copy()

    jobs["submit_time_s"] = pd.to_numeric(
        jobs["submit_time_s"],
        errors="raise",
    )
    jobs["mapped_num_gpus"] = pd.to_numeric(
        jobs["mapped_num_gpus"],
        errors="raise",
    ).astype(int)
    jobs["mapped_solo_runtime_s"] = pd.to_numeric(
        jobs["mapped_solo_runtime_s"],
        errors="raise",
    )

    if (jobs["submit_time_s"] < 0).any():
        raise ValueError("submit_time_s must be nonnegative")

    if (jobs["mapped_num_gpus"] <= 0).any():
        raise ValueError("mapped_num_gpus must be positive")

    if (jobs["mapped_num_gpus"] > server_gpu_count).any():
        unsupported = sorted(
            jobs.loc[
                jobs["mapped_num_gpus"] > server_gpu_count,
                "mapped_num_gpus",
            ].unique()
        )
        raise ValueError(
            "Jobs request more GPUs than the server provides: "
            f"{unsupported}"
        )

    if (jobs["mapped_solo_runtime_s"] <= 0).any():
        raise ValueError(
            "mapped_solo_runtime_s must be positive"
        )

    jobs["_original_index"] = range(len(jobs))
    jobs = jobs.sort_values(
        ["submit_time_s", "_original_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Heap entries are (free_time, gpu_id).
    gpu_heap = [
        (0.0, gpu_id)
        for gpu_id in range(server_gpu_count)
    ]
    heapq.heapify(gpu_heap)

    rows = []

    for simulation_index, job in jobs.iterrows():
        required_gpus = int(job["mapped_num_gpus"])

        selected = [
            heapq.heappop(gpu_heap)
            for _ in range(required_gpus)
        ]

        selected_gpu_ids = sorted(
            gpu_id for _, gpu_id in selected
        )

        earliest_gpu_time = max(
            free_time for free_time, _ in selected
        )

        start_time_s = max(
            float(job["submit_time_s"]),
            earliest_gpu_time,
        )

        end_time_s = (
            start_time_s
            + float(job["mapped_solo_runtime_s"])
        )

        for _, gpu_id in selected:
            heapq.heappush(
                gpu_heap,
                (end_time_s, gpu_id),
            )

        row = job.drop(labels=["_original_index"]).to_dict()
        row.update(
            {
                "simulation_job_index": int(
                    simulation_index
                ),
                "exclusive_gpu_ids": ",".join(
                    str(gpu_id)
                    for gpu_id in selected_gpu_ids
                ),
                "exclusive_start_time_s": start_time_s,
                "exclusive_end_time_s": end_time_s,
                "exclusive_waiting_time_s": (
                    start_time_s
                    - float(job["submit_time_s"])
                ),
                "exclusive_jct_s": (
                    end_time_s
                    - float(job["submit_time_s"])
                ),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)

    first_submit = float(result["submit_time_s"].min())
    final_completion = float(
        result["exclusive_end_time_s"].max()
    )

    summary = {
        "scheduling_model": (
            "arrival_order_earliest_available_exclusive_gpus"
        ),
        "server_gpu_count": int(server_gpu_count),
        "job_count": int(len(result)),
        "arrival_span_s": float(
            result["submit_time_s"].max()
            - first_submit
        ),
        "makespan_s": float(
            final_completion - first_submit
        ),
        "waiting_mean_s": float(
            result["exclusive_waiting_time_s"].mean()
        ),
        "waiting_p50_s": float(
            result["exclusive_waiting_time_s"].quantile(0.50)
        ),
        "waiting_p95_s": float(
            result["exclusive_waiting_time_s"].quantile(0.95)
        ),
        "waiting_p99_s": float(
            result["exclusive_waiting_time_s"].quantile(0.99)
        ),
        "jct_mean_s": float(
            result["exclusive_jct_s"].mean()
        ),
        "jct_p50_s": float(
            result["exclusive_jct_s"].quantile(0.50)
        ),
        "jct_p95_s": float(
            result["exclusive_jct_s"].quantile(0.95)
        ),
        "jct_p99_s": float(
            result["exclusive_jct_s"].quantile(0.99)
        ),
    }

    return result, summary
