from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "workload_id",
    "run_id",
    "spec_path",
    "gpu_count",
    "end_to_end_time_s",
    "exit_code",
    "gpu_memory_requirement_mib",
}


def load_profile_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    return df


def runtime_class(num_gpus: int, solo_runtime_s: float) -> str:
    if num_gpus == 2:
        return "heavy_2gpu"

    if num_gpus == 1 and solo_runtime_s < 600:
        return "light"

    if num_gpus == 1:
        return "medium_heavy"

    raise ValueError(f"Unsupported GPU count: {num_gpus}")



def normalize_task_path(spec_path: str, task_root: Path | None) -> str:
    path = Path(spec_path)

    if task_root is None:
        return path.as_posix()

    task_root = task_root.resolve()
    resolved = path.resolve()

    try:
        return resolved.relative_to(task_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Task path {resolved} is outside task root {task_root}"
        ) from exc


def build_catalog(
    profile_paths: list[Path],
    *,
    task_root: Path | None = None,
) -> pd.DataFrame:
    if not profile_paths:
        raise ValueError("At least one profile CSV is required")

    profiles = pd.concat(
        [load_profile_csv(path) for path in profile_paths],
        ignore_index=True,
    )

    profiles = profiles.loc[profiles["exit_code"] == 0].copy()

    if profiles.empty:
        raise ValueError("No successful profiling rows were found")

    profiles["gpu_count"] = pd.to_numeric(
        profiles["gpu_count"], errors="raise"
    ).astype(int)

    profiles["end_to_end_time_s"] = pd.to_numeric(
        profiles["end_to_end_time_s"], errors="raise"
    )

    profiles["gpu_memory_requirement_mib"] = pd.to_numeric(
        profiles["gpu_memory_requirement_mib"], errors="coerce"
    )

    profiles["task_path"] = profiles["spec_path"].map(
        lambda value: normalize_task_path(str(value), task_root)
    )

    profiles["runtime_class"] = profiles.apply(
        lambda row: runtime_class(
            int(row["gpu_count"]),
            float(row["end_to_end_time_s"]),
        ),
        axis=1,
    )


    profiles["declared_memory_requirement_mib"] = pd.to_numeric(
        profiles.get(
            "gpu_memory_requirement_mib",
            pd.Series(index=profiles.index, dtype="float64"),
        ),
        errors="coerce",
    )

    single_peak = pd.to_numeric(
        profiles.get(
            "gpu_memory_peak_full_mib",
            pd.Series(index=profiles.index, dtype="float64"),
        ),
        errors="coerce",
    )

    gpu_a_peak = pd.to_numeric(
        profiles.get(
            "gpu_memory_peak_full_mib_gpu_a",
            pd.Series(index=profiles.index, dtype="float64"),
        ),
        errors="coerce",
    )

    gpu_b_peak = pd.to_numeric(
        profiles.get(
            "gpu_memory_peak_full_mib_gpu_b",
            pd.Series(index=profiles.index, dtype="float64"),
        ),
        errors="coerce",
    )

    multi_total = pd.to_numeric(
        profiles.get(
            "gpu_memory_peak_full_mib_sum",
            pd.Series(index=profiles.index, dtype="float64"),
        ),
        errors="coerce",
    )

    profiles["measured_peak_memory_per_gpu_mib"] = single_peak

    multi_gpu = profiles["gpu_count"] > 1
    profiles.loc[multi_gpu, "measured_peak_memory_per_gpu_mib"] = (
        pd.concat([gpu_a_peak, gpu_b_peak], axis=1)
        .max(axis=1, skipna=True)
        .loc[multi_gpu]
    )

    profiles["measured_peak_memory_total_mib"] = single_peak
    profiles.loc[multi_gpu, "measured_peak_memory_total_mib"] = (
        multi_total.loc[multi_gpu]
    )


    catalog = profiles.rename(
        columns={
            "gpu_count": "num_gpus",
            "end_to_end_time_s": "solo_runtime_s",
            "run_id": "source_run_id",
        }
    )[
        [
            "workload_id",
            "task_path",
            "num_gpus",
            "solo_runtime_s",
            "declared_memory_requirement_mib",
            "measured_peak_memory_per_gpu_mib",
            "measured_peak_memory_total_mib",
            "source_run_id",
        ]
    ]

    duplicates = catalog["workload_id"].duplicated(keep=False)
    if duplicates.any():
        duplicated_ids = sorted(
            catalog.loc[duplicates, "workload_id"].unique()
        )
        raise ValueError(
            f"Duplicate workload IDs found: {duplicated_ids}"
        )

    return catalog.sort_values("workload_id").reset_index(drop=True)