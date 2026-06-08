from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SOURCE_FORMATS = {"philly", "helios"}
SUPPORTED_GPU_MATCHING = {"exact"}
SUPPORTED_DURATION_MATCHING = {"nearest_quantile"}
SUPPORTED_UNSUPPORTED_GPU_POLICIES = {"filter", "error"}

@dataclass(frozen=True)
class SimulationConfig:
    enabled: bool
    server_gpu_count: int

@dataclass(frozen=True)
class SourceConfig:
    format: str
    trace_path: Path
    cluster_name: str | None = None


@dataclass(frozen=True)
class MappingConfig:
    workload_catalog_path: Path
    seed: int
    num_jobs: int | None
    supported_gpu_counts: tuple[int, ...]
    gpu_demand_matching: str
    duration_matching: str
    unsupported_gpu_policy: str
    preserve_arrivals: bool


@dataclass(frozen=True)
class OutputConfig:
    trace_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class GenerationConfig:
    source: SourceConfig
    mapping: MappingConfig
    simulation: SimulationConfig
    output: OutputConfig


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field '{key}' must be a mapping")
    return value


def _require_value(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if value is None:
        raise ValueError(f"Missing required configuration field: {key}")
    return value


def load_generation_config(path: Path) -> GenerationConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    version = raw.get("version")
    if version != 1:
        raise ValueError(
            f"Unsupported configuration version: {version!r}; expected 1"
        )

    source_raw = _require_mapping(raw, "source")
    mapping_raw = _require_mapping(raw, "mapping")
    output_raw = _require_mapping(raw, "output")

    source_format = str(
        _require_value(source_raw, "format")
    ).lower()

    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(
            f"Unsupported source format: {source_format}"
        )

    gpu_demand_matching = str(
        mapping_raw.get("gpu_demand_matching", "exact")
    )
    if gpu_demand_matching not in SUPPORTED_GPU_MATCHING:
        raise ValueError(
            "Unsupported GPU-demand matching mode: "
            f"{gpu_demand_matching}"
        )

    duration_matching = str(
        mapping_raw.get(
            "duration_matching",
            "nearest_quantile",
        )
    )
    if duration_matching not in SUPPORTED_DURATION_MATCHING:
        raise ValueError(
            f"Unsupported duration matching mode: {duration_matching}"
        )

    unsupported_gpu_policy = str(
        mapping_raw.get("unsupported_gpu_policy", "filter")
    )
    if (
        unsupported_gpu_policy
        not in SUPPORTED_UNSUPPORTED_GPU_POLICIES
    ):
        raise ValueError(
            "Unsupported GPU policy: "
            f"{unsupported_gpu_policy}"
        )

    supported_gpu_counts = tuple(
        int(value)
        for value in mapping_raw.get(
            "supported_gpu_counts",
            [1],
        )
    )

    if not supported_gpu_counts:
        raise ValueError(
            "supported_gpu_counts must contain at least one value"
        )

    if any(value <= 0 for value in supported_gpu_counts):
        raise ValueError(
            "supported_gpu_counts must contain positive integers"
        )

    num_jobs_raw = mapping_raw.get("num_jobs")
    num_jobs = (
        None
        if num_jobs_raw is None
        else int(num_jobs_raw)
    )
    if num_jobs is not None and num_jobs <= 0:
        raise ValueError("num_jobs must be positive when provided")

    simulation_raw = raw.get("simulation", {})

    if not isinstance(simulation_raw, dict):
        raise ValueError(
            "Configuration field 'simulation' must be a mapping"
        )

    simulation_enabled = bool(
        simulation_raw.get("enabled", True)
    )

    server_gpu_count = int(
        simulation_raw.get("server_gpu_count", 1)
    )

    if server_gpu_count <= 0:
        raise ValueError(
            "simulation.server_gpu_count must be positive"
        )

    return GenerationConfig(
        source=SourceConfig(
            format=source_format,
            trace_path=Path(
                _require_value(source_raw, "trace_path")
            ),
            cluster_name=(
                None
                if source_raw.get("cluster_name") is None
                else str(source_raw["cluster_name"])
            ),
        ),
        mapping=MappingConfig(
            workload_catalog_path=Path(
                _require_value(
                    mapping_raw,
                    "workload_catalog_path",
                )
            ),
            seed=int(mapping_raw.get("seed", 42)),
            num_jobs=num_jobs,
            supported_gpu_counts=supported_gpu_counts,
            gpu_demand_matching=gpu_demand_matching,
            duration_matching=duration_matching,
            unsupported_gpu_policy=unsupported_gpu_policy,
            preserve_arrivals=bool(
                mapping_raw.get("preserve_arrivals", True)
            ),
        ),
        output=OutputConfig(
            trace_path=Path(
                _require_value(output_raw, "trace_path")
            ),
            manifest_path=Path(
                _require_value(output_raw, "manifest_path")
            ),
        ),
        simulation=SimulationConfig(
            enabled=simulation_enabled,
            server_gpu_count=server_gpu_count
        )
    )