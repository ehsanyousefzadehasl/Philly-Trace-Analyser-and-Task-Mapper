from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

import json
from pathlib import Path


REQUIRED_COLUMNS = {
    "source_trace",
    "source_cluster",
    "source_job_id",
    "source_duration_s",
    "source_queue_wait_s",
    "source_num_gpus",
    "source_multi_node",
    "submit_time_s",
}


def _single_unique_value(
    series: pd.Series,
    *,
    field_name: str,
) -> str:
    values = series.dropna().astype(str).unique()

    if len(values) != 1:
        raise ValueError(
            f"{field_name} must contain exactly one value; "
            f"found {values.tolist()}"
        )

    return str(values[0])


def _metric_statistics(
    series: pd.Series,
    *,
    prefix: str,
) -> dict[str, float | None]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            f"{prefix}_mean_s": None,
            f"{prefix}_p50_s": None,
            f"{prefix}_p90_s": None,
            f"{prefix}_p95_s": None,
            f"{prefix}_p99_s": None,
            f"{prefix}_max_s": None,
        }

    return {
        f"{prefix}_mean_s": float(values.mean()),
        f"{prefix}_p50_s": float(values.quantile(0.50)),
        f"{prefix}_p90_s": float(values.quantile(0.90)),
        f"{prefix}_p95_s": float(values.quantile(0.95)),
        f"{prefix}_p99_s": float(values.quantile(0.99)),
        f"{prefix}_max_s": float(values.max()),
    }


def summarize_source_jobs(
    jobs: pd.DataFrame,
    *,
    supported_gpu_counts: Iterable[int] = (1, 2),
) -> dict:
    missing = REQUIRED_COLUMNS - set(jobs.columns)
    if missing:
        raise ValueError(
            "Normalized source jobs are missing required columns: "
            f"{sorted(missing)}"
        )

    if jobs.empty:
        raise ValueError("Normalized source jobs must not be empty")

    supported_counts = tuple(
        sorted({int(value) for value in supported_gpu_counts})
    )

    if not supported_counts:
        raise ValueError(
            "supported_gpu_counts must contain at least one value"
        )

    if any(value <= 0 for value in supported_counts):
        raise ValueError(
            "supported_gpu_counts must contain positive integers"
        )

    frame = jobs.copy()

    frame["source_num_gpus"] = pd.to_numeric(
        frame["source_num_gpus"],
        errors="raise",
    ).astype(int)

    frame["source_duration_s"] = pd.to_numeric(
        frame["source_duration_s"],
        errors="raise",
    )

    frame["source_queue_wait_s"] = pd.to_numeric(
        frame["source_queue_wait_s"],
        errors="raise",
    )

    frame["submit_time_s"] = pd.to_numeric(
        frame["submit_time_s"],
        errors="raise",
    )

    if (frame["source_num_gpus"] <= 0).any():
        raise ValueError("source_num_gpus must be positive")

    if (frame["source_duration_s"] <= 0).any():
        raise ValueError("source_duration_s must be positive")

    if (frame["source_queue_wait_s"] < 0).any():
        raise ValueError("source_queue_wait_s must be nonnegative")

    supported = frame[
        frame["source_num_gpus"].isin(supported_counts)
    ].copy()

    total_jobs = len(frame)
    supported_jobs = len(supported)

    frame["gpu_service_time_s"] = (
        frame["source_num_gpus"]
        * frame["source_duration_s"]
    )
    supported["gpu_service_time_s"] = (
        supported["source_num_gpus"]
        * supported["source_duration_s"]
    )

    total_gpu_service_time_s = float(
        frame["gpu_service_time_s"].sum()
    )
    supported_gpu_service_time_s = float(
        supported["gpu_service_time_s"].sum()
    )

    summary = {
        "source_trace": _single_unique_value(
            frame["source_trace"],
            field_name="source_trace",
        ),
        "source_cluster": _single_unique_value(
            frame["source_cluster"],
            field_name="source_cluster",
        ),
        "valid_job_count": int(total_jobs),
        "supported_gpu_counts": list(supported_counts),
        "supported_job_count": int(supported_jobs),
        "unsupported_job_count": int(total_jobs - supported_jobs),
        "supported_job_fraction": float(
            supported_jobs / total_jobs
        ),
        "single_gpu_job_count": int(
            (frame["source_num_gpus"] == 1).sum()
        ),
        "single_gpu_job_fraction": float(
            (frame["source_num_gpus"] == 1).mean()
        ),
        "two_gpu_job_count": int(
            (frame["source_num_gpus"] == 2).sum()
        ),
        "two_gpu_job_fraction": float(
            (frame["source_num_gpus"] == 2).mean()
        ),
        "multi_node_job_count": int(
            frame["source_multi_node"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "multi_node_job_fraction": float(
            frame["source_multi_node"]
            .fillna(False)
            .astype(bool)
            .mean()
        ),
        "arrival_span_s": float(
            frame["submit_time_s"].max()
            - frame["submit_time_s"].min()
        ),
        "total_gpu_service_time_s": (
            total_gpu_service_time_s
        ),
        "supported_gpu_service_time_s": (
            supported_gpu_service_time_s
        ),
        "supported_gpu_service_time_fraction": (
            supported_gpu_service_time_s
            / total_gpu_service_time_s
            if total_gpu_service_time_s > 0
            else None
        ),
    }

    summary.update(
        _metric_statistics(
            frame["source_duration_s"],
            prefix="all_duration",
        )
    )
    summary.update(
        _metric_statistics(
            frame["source_queue_wait_s"],
            prefix="all_queue_wait",
        )
    )
    summary.update(
        _metric_statistics(
            supported["source_duration_s"],
            prefix="supported_duration",
        )
    )
    summary.update(
        _metric_statistics(
            supported["source_queue_wait_s"],
            prefix="supported_queue_wait",
        )
    )

    return summary


def gpu_demand_distribution(
    jobs: pd.DataFrame,
) -> pd.DataFrame:
    if "source_num_gpus" not in jobs:
        raise ValueError(
            "Normalized source jobs are missing source_num_gpus"
        )

    counts = (
        jobs["source_num_gpus"]
        .value_counts()
        .sort_index()
        .rename_axis("source_num_gpus")
        .reset_index(name="job_count")
    )

    total = int(counts["job_count"].sum())

    counts["job_fraction"] = (
        counts["job_count"] / total
        if total > 0
        else 0.0
    )

    return counts


def write_source_summary_artifacts(
    *,
    jobs: pd.DataFrame,
    supported_gpu_counts: Iterable[int],
    output_dir: Path,
) -> tuple[dict, Path, Path]:
    summary = summarize_source_jobs(
        jobs,
        supported_gpu_counts=supported_gpu_counts,
    )
    distribution = gpu_demand_distribution(jobs)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "source_summary.json"
    distribution_path = (
        output_dir / "gpu_demand_distribution.csv"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    distribution.to_csv(distribution_path, index=False)

    return summary, summary_path, distribution_path