from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "job_id",
    "gpu_num",
    "node_num",
    "state",
    "submit_time",
    "start_time",
    "end_time",
    "duration",
    "queue",
}


def load_helios_jobs(
    path: Path,
    *,
    cluster_name: str,
) -> pd.DataFrame:
    cluster_name = str(cluster_name).strip().lower()
    if not cluster_name:
        raise ValueError("cluster_name must not be empty")

    df = pd.read_csv(
        path,
        usecols=sorted(REQUIRED_COLUMNS),
    )

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    df["submitted_at"] = pd.to_datetime(
        df["submit_time"],
        errors="coerce",
    )
    df["source_start_at"] = pd.to_datetime(
        df["start_time"],
        errors="coerce",
    )
    df["source_end_at"] = pd.to_datetime(
        df["end_time"],
        errors="coerce",
    )

    df["gpu_num"] = pd.to_numeric(
        df["gpu_num"],
        errors="coerce",
    )
    df["node_num"] = pd.to_numeric(
        df["node_num"],
        errors="coerce",
    )
    df["duration"] = pd.to_numeric(
        df["duration"],
        errors="coerce",
    )
    df["queue"] = pd.to_numeric(
        df["queue"],
        errors="coerce",
    )

    valid = (
        df["state"].astype(str).eq("COMPLETED")
        & df["submitted_at"].notna()
        & df["source_start_at"].notna()
        & df["source_end_at"].notna()
        & df["gpu_num"].gt(0)
        & df["node_num"].gt(0)
        & df["duration"].gt(0)
        & df["queue"].ge(0)
    )

    df = df.loc[valid].copy()

    if df.empty:
        raise ValueError(
            f"No valid completed GPU jobs found in {path}"
        )

    timestamp_duration_s = (
        df["source_end_at"] - df["source_start_at"]
    ).dt.total_seconds()

    timestamp_queue_s = (
        df["source_start_at"] - df["submitted_at"]
    ).dt.total_seconds()

    duration_mismatch = (
        timestamp_duration_s - df["duration"]
    ).abs() > 1.0

    queue_mismatch = (
        timestamp_queue_s - df["queue"]
    ).abs() > 1.0

    if duration_mismatch.any():
        raise ValueError(
            f"{path} contains "
            f"{int(duration_mismatch.sum())} duration mismatches"
        )

    if queue_mismatch.any():
        raise ValueError(
            f"{path} contains "
            f"{int(queue_mismatch.sum())} queue-time mismatches"
        )

    result = pd.DataFrame(
        {
            "source_trace": "helios",
            "source_cluster": cluster_name,
            "source_job_id": df["job_id"].astype(str),
            "source_status": df["state"].astype(str),
            "submitted_at": df["submitted_at"],
            "source_start_at": df["source_start_at"],
            "source_end_at": df["source_end_at"],
            "source_duration_s": df["duration"].astype(float),
            "source_queue_wait_s": df["queue"].astype(float),
            "source_num_gpus": df["gpu_num"].astype(int),
            "source_node_count": df["node_num"].astype(int),
            "source_multi_node": df["node_num"].gt(1),
        }
    )

    result = result.sort_values(
        ["submitted_at", "source_job_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if result["source_job_id"].duplicated().any():
        duplicates = result.loc[
            result["source_job_id"].duplicated(keep=False),
            "source_job_id",
        ].unique()

        raise ValueError(
            "Duplicate Helios job IDs found: "
            f"{sorted(duplicates)[:10]}"
        )

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