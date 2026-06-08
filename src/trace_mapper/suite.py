from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from trace_mapper.sources.helios import load_helios_jobs
from trace_mapper.sources.philly import load_philly_jobs
from trace_mapper.summary import write_source_summary_artifacts


SUPPORTED_SOURCE_FORMATS = {"philly", "helios"}


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (base_dir / path).resolve()


def _require_mapping(
    data: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = data.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"Configuration field '{key}' must be a mapping"
        )

    return value


def load_suite_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(
            "Suite configuration root must be a mapping"
        )

    if raw.get("version") != 1:
        raise ValueError(
            "Suite configuration version must be 1"
        )

    root_dir_value = str(raw.get("root_dir", "."))
    base_dir = _resolve_path(path.parent, root_dir_value)

    supported_gpu_counts = tuple(
        sorted(
            {
                int(value)
                for value in raw.get(
                    "supported_gpu_counts",
                    [1, 2],
                )
            }
        )
    )

    if not supported_gpu_counts:
        raise ValueError(
            "supported_gpu_counts must not be empty"
        )

    if any(value <= 0 for value in supported_gpu_counts):
        raise ValueError(
            "supported_gpu_counts must contain positive integers"
        )

    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError(
            "sources must contain at least one source"
        )

    sources = []
    seen_names = set()

    for source_raw in sources_raw:
        if not isinstance(source_raw, dict):
            raise ValueError(
                "Every source entry must be a mapping"
            )

        name = str(source_raw.get("name", "")).strip().lower()
        source_format = str(
            source_raw.get("format", "")
        ).strip().lower()
        trace_path_raw = source_raw.get("trace_path")

        if not name:
            raise ValueError(
                "Every source must have a nonempty name"
            )

        if name in seen_names:
            raise ValueError(
                f"Duplicate source name: {name}"
            )
        seen_names.add(name)

        if source_format not in SUPPORTED_SOURCE_FORMATS:
            raise ValueError(
                f"Unsupported source format: {source_format}"
            )

        if trace_path_raw is None:
            raise ValueError(
                f"Source '{name}' is missing trace_path"
            )

        sources.append(
            {
                "name": name,
                "format": source_format,
                "trace_path": _resolve_path(
                    base_dir,
                    str(trace_path_raw),
                ),
                "cluster_name": str(
                    source_raw.get("cluster_name", name)
                ).strip().lower(),
            }
        )

    output_raw = _require_mapping(raw, "output")

    artifact_dir_raw = output_raw.get("artifact_dir")
    markdown_path_raw = output_raw.get("markdown_path")

    if artifact_dir_raw is None:
        raise ValueError(
            "output.artifact_dir is required"
        )

    if markdown_path_raw is None:
        raise ValueError(
            "output.markdown_path is required"
        )

    return {
        "supported_gpu_counts": supported_gpu_counts,
        "sources": sources,
        "artifact_dir": _resolve_path(
            base_dir,
            str(artifact_dir_raw),
        ),
        "markdown_path": _resolve_path(
            base_dir,
            str(markdown_path_raw),
        ),
    }


def _load_source(source: dict[str, Any]) -> pd.DataFrame:
    trace_path = source["trace_path"]

    if not trace_path.is_file():
        raise FileNotFoundError(
            f"Source trace does not exist: {trace_path}"
        )

    if source["format"] == "philly":
        return load_philly_jobs(trace_path)

    return load_helios_jobs(
        trace_path,
        cluster_name=source["cluster_name"],
    )

def _write_suite_figures(
    summary_df: pd.DataFrame,
    *,
    figure_dir: Path,
) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = summary_df.copy()
    frame["label"] = (
        frame["source_cluster"]
        .astype(str)
        .str.title()
    )

    coverage_path = figure_dir / "supported_coverage.png"

    x = list(range(len(frame)))
    width = 0.38

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        [value - width / 2 for value in x],
        frame["supported_job_fraction"] * 100.0,
        width=width,
        label="Job-count coverage",
    )
    axis.bar(
        [value + width / 2 for value in x],
        frame["supported_gpu_service_time_fraction"] * 100.0,
        width=width,
        label="GPU-service coverage",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(frame["label"])
    axis.set_ylabel("Coverage (%)")
    axis.set_title("Coverage of 1–2 GPU jobs")
    axis.legend()
    figure.tight_layout()
    figure.savefig(coverage_path, dpi=200)
    plt.close(figure)

    gpu_mix_path = figure_dir / "gpu_demand_mix.png"

    one_gpu = frame["single_gpu_job_fraction"] * 100.0
    two_gpu = frame["two_gpu_job_fraction"] * 100.0
    other_gpu = 100.0 - one_gpu - two_gpu

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(frame["label"], one_gpu, label="1 GPU")
    axis.bar(
        frame["label"],
        two_gpu,
        bottom=one_gpu,
        label="2 GPUs",
    )
    axis.bar(
        frame["label"],
        other_gpu,
        bottom=one_gpu + two_gpu,
        label="More than 2 GPUs",
    )
    axis.set_ylabel("Jobs (%)")
    axis.set_title("GPU-demand composition")
    axis.legend()
    figure.tight_layout()
    figure.savefig(gpu_mix_path, dpi=200)
    plt.close(figure)

    runtime_path = figure_dir / "runtime_percentiles.png"

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        [value - width / 2 for value in x],
        frame["all_duration_p50_s"] / 3600.0,
        width=width,
        label="p50",
    )
    axis.bar(
        [value + width / 2 for value in x],
        frame["all_duration_p95_s"] / 3600.0,
        width=width,
        label="p95",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(frame["label"])
    axis.set_ylabel("Runtime (hours)")
    axis.set_title("Source-job runtime percentiles")
    axis.legend()
    figure.tight_layout()
    figure.savefig(runtime_path, dpi=200)
    plt.close(figure)

    queue_path = figure_dir / "queue_percentiles.png"

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        [value - width / 2 for value in x],
        frame["all_queue_wait_p50_s"] / 60.0,
        width=width,
        label="p50",
    )
    axis.bar(
        [value + width / 2 for value in x],
        frame["all_queue_wait_p95_s"] / 60.0,
        width=width,
        label="p95",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(frame["label"])
    axis.set_ylabel("Queue time (minutes)")
    axis.set_title("Source-job queue-time percentiles")
    axis.legend()
    figure.tight_layout()
    figure.savefig(queue_path, dpi=200)
    plt.close(figure)

    return {
        "coverage_path": coverage_path,
        "gpu_mix_path": gpu_mix_path,
        "runtime_path": runtime_path,
        "queue_path": queue_path,
    }

def _markdown_report(rows: list[dict]) -> str:
    lines = [
        "# Production Trace Characterization",
        "",
        "This report is generated by `trace-mapper summarize-suite`. "
        "It summarizes normalized, successfully completed GPU jobs. "
        "The supported subset contains jobs whose GPU demands match "
        "the configured target-server scope.",
        "",
        "| Trace | Valid jobs | Supported jobs | Job coverage | "
        "GPU-service coverage | 1-GPU jobs | 2-GPU jobs | "
        "Runtime p50 | Runtime p95 | Queue p50 | Queue p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        service_fraction = row[
            "supported_gpu_service_time_fraction"
        ]
        service_text = (
            f"{service_fraction:.2%}"
            if service_fraction is not None
            else "N/A"
        )

        lines.append(
            f"| {row['source_cluster'].title()} | "
            f"{row['valid_job_count']:,} | "
            f"{row['supported_job_count']:,} | "
            f"{row['supported_job_fraction']:.2%} | "
            f"{service_text} | "
            f"{row['single_gpu_job_fraction']:.2%} | "
            f"{row['two_gpu_job_fraction']:.2%} | "
            f"{row['all_duration_p50_s']:.0f} s | "
            f"{row['all_duration_p95_s']:.0f} s | "
            f"{row['all_queue_wait_p50_s']:.0f} s | "
            f"{row['all_queue_wait_p95_s']:.0f} s |"
        )

    lines.extend(
        [
            "",
            "## Supported Testbed Coverage",
            "",
            "![Coverage of supported GPU demands]"
            "(trace_characterization/supported_coverage.png)",
            "",
            "Job-count coverage measures the fraction of jobs that request "
            "one or two GPUs. GPU-service coverage weights each job by its "
            "GPU demand and runtime, showing how much total GPU work the "
            "supported subset represents.",
            "",
            "## GPU-Demand Composition",
            "",
            "![GPU-demand composition]"
            "(trace_characterization/gpu_demand_mix.png)",
            "",
            "## Runtime Distribution",
            "",
            "![Runtime percentiles]"
            "(trace_characterization/runtime_percentiles.png)",
            "",
            "## Queue-Time Distribution",
            "",
            "![Queue-time percentiles]"
            "(trace_characterization/queue_percentiles.png)",
            "",
            "Raw production traces are not distributed with this "
            "repository. See `data/README.md` for their sources, "
            "licenses, and expected local paths.",
            "",
        ]
    )

    return "\n".join(lines)


def run_summary_suite(config_path: Path) -> dict[str, Path]:
    config = load_suite_config(config_path)

    artifact_dir = config["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for source in config["sources"]:
        jobs = _load_source(source)

        summary, _, _ = write_source_summary_artifacts(
            jobs=jobs,
            supported_gpu_counts=config[
                "supported_gpu_counts"
            ],
            output_dir=artifact_dir / source["name"],
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["source_trace", "source_cluster"],
        kind="mergesort",
    )

    suite_csv_path = artifact_dir / "trace_suite_summary.csv"
    suite_json_path = artifact_dir / "trace_suite_summary.json"

    summary_df.to_csv(suite_csv_path, index=False)

    suite_json_path.write_text(
        json.dumps(
            summary_df.to_dict(orient="records"),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown_path = config["markdown_path"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _markdown_report(
            summary_df.to_dict(orient="records")
        ),
        encoding="utf-8",
    )

    figure_dir = (
        markdown_path.parent
        / markdown_path.stem
    )

    figure_paths = _write_suite_figures(
        summary_df,
        figure_dir=figure_dir,
    )

    return {
        "suite_csv_path": suite_csv_path,
        "suite_json_path": suite_json_path,
        "markdown_path": markdown_path,
        **figure_paths,
    }
