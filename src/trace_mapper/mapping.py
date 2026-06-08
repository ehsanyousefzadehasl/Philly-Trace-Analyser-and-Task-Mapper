from __future__ import annotations

import pandas as pd


SOURCE_REQUIRED_COLUMNS = {
    "source_trace",
    "source_cluster",
    "source_job_id",
    "source_duration_s",
    "source_num_gpus",
    "submit_time_s",
    "source_interarrival_s",
}

CATALOG_REQUIRED_COLUMNS = {
    "workload_id",
    "task_path",
    "num_gpus",
    "solo_runtime_s",
}


def _validate_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    frame_name: str,
) -> None:
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def _runtime_quantiles(
    frame: pd.DataFrame,
    *,
    duration_column: str,
    tie_break_column: str,
) -> pd.Series:
    ordered = frame.sort_values(
        [duration_column, tie_break_column],
        kind="mergesort",
    )

    count = len(ordered)

    if count == 1:
        quantiles = pd.Series(
            [0.5],
            index=ordered.index,
            dtype="float64",
        )
    else:
        quantiles = pd.Series(
            [
                index / (count - 1)
                for index in range(count)
            ],
            index=ordered.index,
            dtype="float64",
        )

    return quantiles.reindex(frame.index)


def map_jobs_to_workloads(
    source_jobs: pd.DataFrame,
    workload_catalog: pd.DataFrame,
    *,
    supported_gpu_counts: tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    _validate_columns(
        source_jobs,
        SOURCE_REQUIRED_COLUMNS,
        frame_name="source_jobs",
    )
    _validate_columns(
        workload_catalog,
        CATALOG_REQUIRED_COLUMNS,
        frame_name="workload_catalog",
    )

    supported_gpu_counts = tuple(
        sorted({int(value) for value in supported_gpu_counts})
    )

    if not supported_gpu_counts:
        raise ValueError(
            "supported_gpu_counts must contain at least one value"
        )

    if any(value <= 0 for value in supported_gpu_counts):
        raise ValueError(
            "supported_gpu_counts must contain positive integers"
        )

    jobs = source_jobs.copy()
    catalog = workload_catalog.copy()

    jobs["source_num_gpus"] = pd.to_numeric(
        jobs["source_num_gpus"],
        errors="raise",
    ).astype(int)

    jobs["source_duration_s"] = pd.to_numeric(
        jobs["source_duration_s"],
        errors="raise",
    )

    catalog["num_gpus"] = pd.to_numeric(
        catalog["num_gpus"],
        errors="raise",
    ).astype(int)

    catalog["solo_runtime_s"] = pd.to_numeric(
        catalog["solo_runtime_s"],
        errors="raise",
    )

    jobs = jobs.loc[
        jobs["source_num_gpus"].isin(supported_gpu_counts)
    ].copy()

    catalog = catalog.loc[
        catalog["num_gpus"].isin(supported_gpu_counts)
    ].copy()

    if jobs.empty:
        raise ValueError(
            "No source jobs remain after GPU-demand filtering"
        )

    if catalog.empty:
        raise ValueError(
            "No workloads remain after GPU-demand filtering"
        )

    source_gpu_counts = set(jobs["source_num_gpus"].unique())
    catalog_gpu_counts = set(catalog["num_gpus"].unique())

    missing_gpu_counts = sorted(
        source_gpu_counts - catalog_gpu_counts
    )

    if missing_gpu_counts:
        raise ValueError(
            "The workload catalog has no entries for GPU counts: "
            f"{missing_gpu_counts}"
        )

    mapped_rows: list[dict] = []

    for gpu_count in sorted(source_gpu_counts):
        group_jobs = jobs.loc[
            jobs["source_num_gpus"] == gpu_count
        ].copy()

        group_catalog = catalog.loc[
            catalog["num_gpus"] == gpu_count
        ].copy()

        group_jobs["source_runtime_quantile"] = (
            _runtime_quantiles(
                group_jobs,
                duration_column="source_duration_s",
                tie_break_column="source_job_id",
            )
        )

        group_catalog["mapped_runtime_quantile"] = (
            _runtime_quantiles(
                group_catalog,
                duration_column="solo_runtime_s",
                tie_break_column="workload_id",
            )
        )

        group_catalog = group_catalog.sort_values(
            [
                "mapped_runtime_quantile",
                "solo_runtime_s",
                "workload_id",
            ],
            kind="mergesort",
        )

        for _, source_job in group_jobs.iterrows():
            distances = (
                group_catalog["mapped_runtime_quantile"]
                - float(
                    source_job["source_runtime_quantile"]
                )
            ).abs()

            nearest_index = distances.idxmin()
            workload = group_catalog.loc[nearest_index]

            row = source_job.to_dict()
            row.update(
                {
                    "mapped_workload_id": workload[
                        "workload_id"
                    ],
                    "task_path": workload["task_path"],
                    "mapped_num_gpus": int(
                        workload["num_gpus"]
                    ),
                    "mapped_solo_runtime_s": float(
                        workload["solo_runtime_s"]
                    ),
                    "mapped_runtime_quantile": float(
                        workload["mapped_runtime_quantile"]
                    ),
                    "runtime_quantile_distance": abs(
                        float(
                            source_job[
                                "source_runtime_quantile"
                            ]
                        )
                        - float(
                            workload[
                                "mapped_runtime_quantile"
                            ]
                        )
                    ),
                }
            )

            mapped_rows.append(row)

    result = pd.DataFrame(mapped_rows)

    result = result.sort_values(
        ["submit_time_s", "source_job_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if not (
        result["source_num_gpus"]
        == result["mapped_num_gpus"]
    ).all():
        raise AssertionError(
            "Mapped GPU demand does not match source GPU demand"
        )

    return result
