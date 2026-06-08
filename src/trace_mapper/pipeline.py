from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml

from trace_mapper.catalog import build_catalog
from trace_mapper.generation import run_generation
from trace_mapper.report import (
    generate_trace_report,
    generate_trace_suite_report,
)
from trace_mapper.suite import run_summary_suite


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()

    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(previous)


def _require_mapping(
    value: object,
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"Pipeline field '{field_name}' must be a mapping"
        )

    return value


def _resolve_path(root_dir: Path, value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (root_dir / path).resolve()


def load_pipeline_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(
            "Pipeline configuration must be a mapping"
        )

    version = int(raw.get("version", 1))

    if version != 1:
        raise ValueError(
            f"Unsupported pipeline configuration version: {version}"
        )

    root_value = raw.get("root_dir", ".")

    root_dir = _resolve_path(
        config_path.parent,
        str(root_value),
    )

    catalog = _require_mapping(
        raw.get("catalog"),
        field_name="catalog",
    )
    characterization = _require_mapping(
        raw.get("characterization"),
        field_name="characterization",
    )
    generation = _require_mapping(
        raw.get("generation"),
        field_name="generation",
    )
    reports = _require_mapping(
        raw.get("reports", {}),
        field_name="reports",
    )
    validation = _require_mapping(
        raw.get("validation", {}),
        field_name="validation",
    )

    return {
        "version": version,
        "config_path": config_path,
        "root_dir": root_dir,
        "catalog": catalog,
        "characterization": characterization,
        "generation": generation,
        "reports": reports,
        "validation": validation,
    }


def _validate_generated_suite(
    *,
    manifest_paths: list[Path],
    catalog_path: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    required_job_count_raw = validation.get(
        "required_job_count"
    )
    required_job_count = (
        int(required_job_count_raw)
        if required_job_count_raw is not None
        else None
    )

    require_two_gpu_job = bool(
        validation.get(
            "require_two_gpu_job_per_trace",
            False,
        )
    )

    require_all_two_gpu_workloads = bool(
        validation.get(
            "require_all_two_gpu_workloads_across_suite",
            False,
        )
    )

    maximum_gpu_deviation = float(
        validation.get(
            "maximum_gpu_fraction_deviation",
            1.0,
        )
    )

    maximum_runtime_deviation = float(
        validation.get(
            "maximum_runtime_cdf_deviation",
            1.0,
        )
    )

    catalog = pd.read_csv(catalog_path)

    required_catalog_columns = {
        "workload_id",
        "num_gpus",
    }
    missing_catalog_columns = (
        required_catalog_columns - set(catalog.columns)
    )

    if missing_catalog_columns:
        raise ValueError(
            "Catalog is missing validation columns: "
            f"{sorted(missing_catalog_columns)}"
        )

    available_two_gpu_workloads = set(
        catalog.loc[
            pd.to_numeric(
                catalog["num_gpus"],
                errors="raise",
            ).astype(int) == 2,
            "workload_id",
        ].astype(str)
    )

    used_two_gpu_workloads: set[str] = set()
    errors: list[str] = []
    trace_rows: list[dict[str, Any]] = []

    for manifest_path in manifest_paths:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        trace_name = str(
            manifest.get("configuration", {}).get(
                "source_cluster",
                manifest_path.stem,
            )
        )

        trace_path = Path(
            manifest["output"]["trace_path"]
        )

        if not trace_path.is_absolute():
            trace_path = (
                manifest_path.parent / trace_path
            ).resolve()

        trace = pd.read_csv(trace_path)

        job_count = len(trace)
        two_gpu_job_count = int(
            (
                pd.to_numeric(
                    trace["mapped_num_gpus"],
                    errors="raise",
                ).astype(int)
                == 2
            ).sum()
        )

        trace_two_gpu_workloads = set(
            trace.loc[
                pd.to_numeric(
                    trace["mapped_num_gpus"],
                    errors="raise",
                ).astype(int) == 2,
                "mapped_workload_id",
            ].astype(str)
        )

        used_two_gpu_workloads.update(
            trace_two_gpu_workloads
        )

        representative = (
            manifest.get("selection", {})
            .get("representativeness")
        )

        if not isinstance(representative, dict):
            errors.append(
                f"{trace_name}: missing representativeness metadata"
            )
            continue

        target = representative["target_profile"]
        selected = representative["selected_profile"]

        gpu_deviations = [
            abs(
                float(
                    selected["gpu_demand_fractions"].get(
                        gpu_count,
                        0.0,
                    )
                )
                - float(
                    target["gpu_demand_fractions"].get(
                        gpu_count,
                        0.0,
                    )
                )
            )
            for gpu_count in sorted(
                set(target["gpu_demand_fractions"])
                | set(selected["gpu_demand_fractions"])
            )
        ]

        runtime_deviations = [
            abs(
                float(selected["runtime_cdf"][bucket])
                - float(target["runtime_cdf"][bucket])
            )
            for bucket in target["runtime_cdf"]
        ]

        max_gpu_deviation = (
            max(gpu_deviations)
            if gpu_deviations
            else 0.0
        )
        max_runtime_deviation = (
            max(runtime_deviations)
            if runtime_deviations
            else 0.0
        )

        if (
            required_job_count is not None
            and job_count != required_job_count
        ):
            errors.append(
                f"{trace_name}: expected "
                f"{required_job_count} jobs, found {job_count}"
            )

        if require_two_gpu_job and two_gpu_job_count < 1:
            errors.append(
                f"{trace_name}: contains no 2-GPU job"
            )

        if max_gpu_deviation > maximum_gpu_deviation:
            errors.append(
                f"{trace_name}: maximum GPU-demand deviation "
                f"{max_gpu_deviation:.4f} exceeds "
                f"{maximum_gpu_deviation:.4f}"
            )

        if (
            max_runtime_deviation
            > maximum_runtime_deviation
        ):
            errors.append(
                f"{trace_name}: maximum runtime-CDF deviation "
                f"{max_runtime_deviation:.4f} exceeds "
                f"{maximum_runtime_deviation:.4f}"
            )

        trace_rows.append(
            {
                "trace": trace_name,
                "job_count": job_count,
                "two_gpu_job_count": two_gpu_job_count,
                "two_gpu_workload_count": len(
                    trace_two_gpu_workloads
                ),
                "maximum_gpu_fraction_deviation": (
                    max_gpu_deviation
                ),
                "maximum_runtime_cdf_deviation": (
                    max_runtime_deviation
                ),
                "representativeness_score": float(
                    representative["total_score"]
                ),
            }
        )

    missing_two_gpu_workloads = sorted(
        available_two_gpu_workloads
        - used_two_gpu_workloads
    )

    if (
        require_all_two_gpu_workloads
        and missing_two_gpu_workloads
    ):
        errors.append(
            "Suite does not cover all 2-GPU workloads; "
            f"missing={missing_two_gpu_workloads}"
        )

    result = {
        "passed": not errors,
        "errors": errors,
        "traces": trace_rows,
        "available_two_gpu_workloads": sorted(
            available_two_gpu_workloads
        ),
        "used_two_gpu_workloads": sorted(
            used_two_gpu_workloads
        ),
        "missing_two_gpu_workloads": (
            missing_two_gpu_workloads
        ),
    }

    if errors:
        formatted = "\n".join(
            f"- {message}" for message in errors
        )

        raise ValueError(
            "Pipeline validation failed:\n"
            f"{formatted}"
        )

    return result


def run_pipeline(config_path: Path) -> dict[str, Any]:
    config = load_pipeline_config(config_path)

    root_dir: Path = config["root_dir"]
    catalog_config = config["catalog"]
    characterization_config = config["characterization"]
    generation_config = config["generation"]
    report_config = config["reports"]
    validation_config = config["validation"]

    profile_values = catalog_config.get("profiles")

    if (
        not isinstance(profile_values, list)
        or not profile_values
    ):
        raise ValueError(
            "catalog.profiles must be a nonempty list"
        )

    catalog_path = _resolve_path(
        root_dir,
        str(catalog_config["output"]),
    )

    profile_paths = [
        _resolve_path(root_dir, str(value))
        for value in profile_values
    ]

    task_root_raw = catalog_config.get("task_root")
    task_root = (
        _resolve_path(root_dir, str(task_root_raw))
        if task_root_raw is not None
        else None
    )

    suite_config_path = _resolve_path(
        root_dir,
        str(characterization_config["config"]),
    )

    generation_values = generation_config.get("configs")

    if (
        not isinstance(generation_values, list)
        or not generation_values
    ):
        raise ValueError(
            "generation.configs must be a nonempty list"
        )

    generation_config_paths = [
        _resolve_path(root_dir, str(value))
        for value in generation_values
    ]

    comparison_output_dir = _resolve_path(
        root_dir,
        str(
            report_config.get(
                "comparison_output_dir",
                "docs/generated_traces",
            )
        ),
    )

    generate_per_trace_reports = bool(
        report_config.get("per_trace", True)
    )

    with _working_directory(root_dir):
        catalog = build_catalog(
            profile_paths,
            task_root=task_root,
        )

        catalog_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        catalog.to_csv(catalog_path, index=False)

        characterization_outputs = run_summary_suite(
            suite_config_path
        )

        generation_outputs = []
        manifest_paths: list[Path] = []

        for generation_path in generation_config_paths:
            outputs = run_generation(generation_path)
            generation_outputs.append(outputs)

            manifest_path = Path(
                outputs["manifest_path"]
            ).resolve()
            manifest_paths.append(manifest_path)

            if generate_per_trace_reports:
                generate_trace_report(
                    manifest_path
                )

        comparison_outputs = (
            generate_trace_suite_report(
                manifest_paths,
                output_dir=comparison_output_dir,
            )
        )

        validation_result = (
            _validate_generated_suite(
                manifest_paths=manifest_paths,
                catalog_path=catalog_path,
                validation=validation_config,
            )
        )

    pipeline_summary_path = (
        comparison_output_dir
        / "pipeline_summary.json"
    )

    pipeline_summary = {
        "pipeline_config": str(
            config["config_path"]
        ),
        "root_dir": str(root_dir),
        "catalog_path": str(catalog_path),
        "catalog_workload_count": int(len(catalog)),
        "manifest_paths": [
            str(path) for path in manifest_paths
        ],
        "validation": validation_result,
    }

    pipeline_summary_path.write_text(
        json.dumps(
            pipeline_summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "catalog_path": catalog_path,
        "characterization_outputs": (
            characterization_outputs
        ),
        "generation_outputs": generation_outputs,
        "comparison_outputs": comparison_outputs,
        "pipeline_summary_path": pipeline_summary_path,
        "validation": validation_result,
    }
