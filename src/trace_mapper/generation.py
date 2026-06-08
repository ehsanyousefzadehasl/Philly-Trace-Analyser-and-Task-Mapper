from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from trace_mapper.config import load_generation_config
from trace_mapper.mapping import map_jobs_to_workloads
from trace_mapper.selection import select_contiguous_job_window
from trace_mapper.sources.helios import load_helios_jobs
from trace_mapper.sources.philly import load_philly_jobs

from trace_mapper.simulation import (
    simulate_exclusive_execution,
)

from trace_mapper.representative import (
    select_representative_job_window,
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def package_version() -> str:
    try:
        return version("dl-trace-mapper")
    except PackageNotFoundError:
        return "unknown"


def load_source_jobs(
    *,
    source_format: str,
    trace_path: Path,
    cluster_name: str | None,
) -> pd.DataFrame:
    if source_format == "philly":
        return load_philly_jobs(trace_path)

    if source_format == "helios":
        if cluster_name is None or not cluster_name.strip():
            raise ValueError(
                "cluster_name is required for Helios traces"
            )

        return load_helios_jobs(
            trace_path,
            cluster_name=cluster_name,
        )

    raise ValueError(
        f"Unsupported source format: {source_format}"
    )


def run_generation(config_path: Path) -> dict[str, object]:
    config = load_generation_config(config_path)

    source_path = config.source.trace_path.resolve()
    catalog_path = (
        config.mapping.workload_catalog_path.resolve()
    )

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source trace does not exist: {source_path}"
        )

    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"Workload catalog does not exist: {catalog_path}"
        )

    if not config.mapping.preserve_arrivals:
        raise ValueError(
            "preserve_arrivals=false is not supported yet"
        )

    source_jobs = load_source_jobs(
        source_format=config.source.format,
        trace_path=source_path,
        cluster_name=config.source.cluster_name,
    )

    supported_gpu_counts = (
        config.mapping.supported_gpu_counts
    )

    minimum_jobs_by_gpu_count = dict(
        config.mapping.minimum_jobs_by_gpu_count
    )

    supported_mask = source_jobs[
        "source_num_gpus"
    ].isin(supported_gpu_counts)

    unsupported_count = int((~supported_mask).sum())

    if (
        config.mapping.unsupported_gpu_policy == "error"
        and unsupported_count > 0
    ):
        raise ValueError(
            f"Source trace contains {unsupported_count} jobs "
            "with unsupported GPU demands"
        )

    eligible_count = int(supported_mask.sum())

    num_jobs = config.mapping.num_jobs
    if num_jobs is None:
        num_jobs = eligible_count

    if config.mapping.selection_method == "representative":
        selected_jobs, selection_metadata = (
            select_representative_job_window(
                source_jobs,
                num_jobs=num_jobs,
                supported_gpu_counts=supported_gpu_counts,
                minimum_jobs_by_gpu_count=(
                    minimum_jobs_by_gpu_count
                ),
            )
        )
    else:
        selected_jobs, selection_metadata = (
            select_contiguous_job_window(
                source_jobs,
                num_jobs=num_jobs,
                supported_gpu_counts=supported_gpu_counts,
                minimum_jobs_by_gpu_count=(
                    minimum_jobs_by_gpu_count
                ),
                seed=config.mapping.seed,
            )
        )

    catalog = pd.read_csv(catalog_path)

    mapped = map_jobs_to_workloads(
        selected_jobs,
        catalog,
        supported_gpu_counts=supported_gpu_counts,
    )

    trace_columns = [
        "submit_time_s",
        "task_path",
        "mapped_workload_id",
        "mapped_num_gpus",
        "mapped_solo_runtime_s",
        "source_trace",
        "source_cluster",
        "source_job_id",
        "source_num_gpus",
        "source_duration_s",
        "source_runtime_quantile",
        "mapped_runtime_quantile",
        "runtime_quantile_distance",
        "source_interarrival_s",
        "source_trace_submit_time_s",
        "source_window_job_index",
    ]

    missing_trace_columns = [
        column
        for column in trace_columns
        if column not in mapped.columns
    ]

    if missing_trace_columns:
        raise ValueError(
            "Mapped trace is missing output columns: "
            f"{missing_trace_columns}"
        )

    trace_output = mapped[trace_columns].copy()

    trace_path = config.output.trace_path
    manifest_path = config.output.manifest_path

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    trace_output.to_csv(trace_path, index=False)

    execution_trace_path = trace_path.with_name(
        f"{trace_path.stem}.execution.csv"
    )

    trace_output[
        ["submit_time_s", "task_path"]
    ].to_csv(
        execution_trace_path,
        index=False,
    )

    exclusive_job_metrics_path = trace_path.with_name(
        f"{trace_path.stem}.exclusive_jobs.csv"
    )

    exclusive_summary_path = trace_path.with_name(
        f"{trace_path.stem}.exclusive_summary.json"
    )

    exclusive_summary = None

    if config.simulation.enabled:
        exclusive_jobs, exclusive_summary = (
            simulate_exclusive_execution(
                trace_output,
                server_gpu_count=(
                    config.simulation.server_gpu_count
                ),
            )
        )

        exclusive_jobs.to_csv(
            exclusive_job_metrics_path,
            index=False,
        )

        exclusive_summary_path.write_text(
            json.dumps(
                exclusive_summary,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    quantile_distance = pd.to_numeric(
        trace_output["runtime_quantile_distance"],
        errors="raise",
    )

    workload_counts = (
        trace_output["mapped_workload_id"]
        .value_counts()
        .sort_index()
    )

    manifest = {
        "manifest_version": 1,
        "tool": {
            "name": "dl-trace-mapper",
            "version": package_version(),
        },
        "configuration": {
            "config_path": str(config_path.resolve()),
            "source_format": config.source.format,
            "source_cluster": config.source.cluster_name,
            "seed": int(config.mapping.seed),
            "selection_method": (
                config.mapping.selection_method
            ),
            "requested_job_count": int(num_jobs),
            "supported_gpu_counts": list(
                supported_gpu_counts
            ),
            "minimum_jobs_by_gpu_count": {
                str(key): int(value)
                for key, value in sorted(
                    minimum_jobs_by_gpu_count.items()
                )
            },
            "gpu_demand_matching": (
                config.mapping.gpu_demand_matching
            ),
            "duration_matching": (
                config.mapping.duration_matching
            ),
            "unsupported_gpu_policy": (
                config.mapping.unsupported_gpu_policy
            ),
            "preserve_arrivals": (
                config.mapping.preserve_arrivals
            ),
            "simulation": {
                "enabled": config.simulation.enabled,
                "server_gpu_count": (
                    config.simulation.server_gpu_count
                ),
            },
        },
        "inputs": {
            "source_trace_path": str(source_path),
            "source_trace_sha256": sha256_file(
                source_path
            ),
            "workload_catalog_path": str(catalog_path),
            "workload_catalog_sha256": sha256_file(
                catalog_path
            ),
        },
        "source_counts": {
            "normalized_job_count": int(
                len(source_jobs)
            ),
            "eligible_job_count": eligible_count,
            "unsupported_job_count": unsupported_count,
        },
        "selection": selection_metadata,
        "mapping": {
            "mapped_job_count": int(len(trace_output)),
            "unique_workload_count": int(
                trace_output[
                    "mapped_workload_id"
                ].nunique()
            ),
            "workload_usage": {
                str(workload_id): int(count)
                for workload_id, count
                in workload_counts.items()
            },
            "runtime_quantile_distance_mean": float(
                quantile_distance.mean()
            ),
            "runtime_quantile_distance_p50": float(
                quantile_distance.quantile(0.50)
            ),
            "runtime_quantile_distance_p95": float(
                quantile_distance.quantile(0.95)
            ),
            "runtime_quantile_distance_max": float(
                quantile_distance.max()
            ),
        },
        "output": {
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": sha256_file(trace_path),
            "execution_trace_path": str(
                execution_trace_path.resolve()
            ),
            "execution_trace_sha256": sha256_file(
                execution_trace_path
            ),
            "exclusive_job_metrics_path": (
                str(exclusive_job_metrics_path.resolve())
                if config.simulation.enabled
                else None
            ),
            "exclusive_summary_path": (
                str(exclusive_summary_path.resolve())
                if config.simulation.enabled
                else None
            ),
        },
        "exclusive_simulation": exclusive_summary,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "trace_path": trace_path,
        "execution_trace_path": execution_trace_path,
        "manifest_path": manifest_path,
        "exclusive_job_metrics_path": (
            exclusive_job_metrics_path
            if config.simulation.enabled
            else None
        ),
        "exclusive_summary_path": (
            exclusive_summary_path
            if config.simulation.enabled
            else None
        ),
        "mapped_job_count": len(trace_output),
    }