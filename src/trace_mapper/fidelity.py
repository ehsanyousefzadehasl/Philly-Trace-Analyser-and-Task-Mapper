from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from trace_mapper.representative import RUNTIME_THRESHOLDS_S


RUNTIME_LABELS = {
    "under_10m": "<10 min",
    "under_1h": "<1 hour",
    "under_6h": "<6 hours",
    "under_1d": "<1 day",
}


def _resolve_manifest_path(
    manifest_path: Path,
    value: str,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (manifest_path.parent / path).resolve()


def _mapped_profile(
    trace: pd.DataFrame,
    *,
    supported_gpu_counts: Iterable[int],
) -> dict:
    required = {
        "mapped_num_gpus",
        "mapped_solo_runtime_s",
        "source_interarrival_s",
    }

    missing = required - set(trace.columns)

    if missing:
        raise ValueError(
            "Mapped trace is missing fidelity columns: "
            f"{sorted(missing)}"
        )

    frame = trace.copy()

    frame["mapped_num_gpus"] = pd.to_numeric(
        frame["mapped_num_gpus"],
        errors="raise",
    ).astype(int)

    frame["mapped_solo_runtime_s"] = pd.to_numeric(
        frame["mapped_solo_runtime_s"],
        errors="raise",
    )

    frame["source_interarrival_s"] = pd.to_numeric(
        frame["source_interarrival_s"],
        errors="raise",
    )

    supported_counts = tuple(
        sorted({int(value) for value in supported_gpu_counts})
    )

    gpu_demand_fractions = {
        str(gpu_count): float(
            (frame["mapped_num_gpus"] == gpu_count).mean()
        )
        for gpu_count in supported_counts
    }

    runtime_cdf = {
        label: float(
            (
                frame["mapped_solo_runtime_s"]
                < threshold_s
            ).mean()
        )
        for label, threshold_s
        in RUNTIME_THRESHOLDS_S.items()
    }

    interarrivals = frame["source_interarrival_s"]

    gpu_service = (
        frame["mapped_num_gpus"]
        * frame["mapped_solo_runtime_s"]
    )

    total_gpu_service = float(gpu_service.sum())

    gpu_service_fractions = {}

    for gpu_count in supported_counts:
        service = float(
            gpu_service.loc[
                frame["mapped_num_gpus"] == gpu_count
            ].sum()
        )

        gpu_service_fractions[str(gpu_count)] = (
            service / total_gpu_service
            if total_gpu_service > 0
            else 0.0
        )

    return {
        "job_count": int(len(frame)),
        "gpu_demand_fractions": gpu_demand_fractions,
        "runtime_cdf": runtime_cdf,
        "interarrival_p50_s": float(
            interarrivals.quantile(0.50)
        ),
        "interarrival_p95_s": float(
            interarrivals.quantile(0.95)
        ),
        "zero_interarrival_fraction": float(
            (interarrivals == 0).mean()
        ),
        "gpu_service_fractions": (
            gpu_service_fractions
        ),
    }


def generate_representative_fidelity_report(
    manifest_paths: list[Path],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    if not manifest_paths:
        raise ValueError(
            "At least one manifest is required"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_rows: list[dict] = []
    runtime_rows: list[dict] = []
    interarrival_rows: list[dict] = []
    score_rows: list[dict] = []
    summary_rows: list[dict] = []

    used_two_gpu: dict[str, set[str]] = {}
    catalog_paths: set[Path] = set()

    for manifest_path in manifest_paths:
        manifest_path = manifest_path.resolve()

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        configuration = manifest["configuration"]
        selection = manifest["selection"]
        representative = selection.get(
            "representativeness"
        )

        if not isinstance(representative, dict):
            raise ValueError(
                f"{manifest_path} does not contain "
                "representative-selection metadata"
            )

        target = representative["target_profile"]
        selected = representative["selected_profile"]

        trace_name = str(
            configuration.get(
                "source_cluster",
                manifest_path.stem,
            )
        ).title()

        supported_gpu_counts = tuple(
            int(value)
            for value in configuration[
                "supported_gpu_counts"
            ]
        )

        trace_path = _resolve_manifest_path(
            manifest_path,
            manifest["output"]["trace_path"],
        )

        trace = pd.read_csv(trace_path)

        mapped = _mapped_profile(
            trace,
            supported_gpu_counts=supported_gpu_counts,
        )

        catalog_path = _resolve_manifest_path(
            manifest_path,
            manifest["inputs"][
                "workload_catalog_path"
            ],
        )
        catalog_paths.add(catalog_path)

        trace_two_gpu = set(
            trace.loc[
                trace["mapped_num_gpus"] == 2,
                "mapped_workload_id",
            ].astype(str)
        )

        for workload_id in trace_two_gpu:
            used_two_gpu.setdefault(
                workload_id,
                set(),
            ).add(trace_name)

        gpu_deltas = []
        runtime_deltas = []
        mapped_runtime_deltas = []

        for gpu_count in supported_gpu_counts:
            key = str(gpu_count)

            target_value = float(
                target[
                    "gpu_demand_fractions"
                ].get(key, 0.0)
            )
            selected_value = float(
                selected[
                    "gpu_demand_fractions"
                ].get(key, 0.0)
            )
            mapped_value = float(
                mapped[
                    "gpu_demand_fractions"
                ].get(key, 0.0)
            )

            gpu_rows.append(
                {
                    "trace": trace_name,
                    "gpu_count": gpu_count,
                    "full_eligible": target_value,
                    "selected_window": selected_value,
                    "mapped_trace": mapped_value,
                }
            )

            gpu_deltas.append(
                abs(selected_value - target_value)
            )

        for bucket in RUNTIME_THRESHOLDS_S:
            target_value = float(
                target["runtime_cdf"][bucket]
            )
            selected_value = float(
                selected["runtime_cdf"][bucket]
            )
            mapped_value = float(
                mapped["runtime_cdf"][bucket]
            )

            runtime_rows.append(
                {
                    "trace": trace_name,
                    "bucket": bucket,
                    "bucket_label": (
                        RUNTIME_LABELS[bucket]
                    ),
                    "full_eligible": target_value,
                    "selected_window": selected_value,
                    "mapped_trace": mapped_value,
                }
            )

            runtime_deltas.append(
                abs(selected_value - target_value)
            )
            mapped_runtime_deltas.append(
                abs(mapped_value - target_value)
            )

        for metric in [
            "interarrival_p50_s",
            "interarrival_p95_s",
            "zero_interarrival_fraction",
        ]:
            interarrival_rows.append(
                {
                    "trace": trace_name,
                    "metric": metric,
                    "full_eligible": float(
                        target[metric]
                    ),
                    "selected_window": float(
                        selected[metric]
                    ),
                    # Mapping preserves selected arrivals.
                    "mapped_trace": float(
                        selected[metric]
                    ),
                }
            )

        component_values = representative["components"]

        expected_components = [
            "gpu_demand_distance",
            "runtime_cdf_distance",
            "interarrival_distance",
            "burst_fraction_distance",
            "gpu_service_distance",
        ]

        for component in expected_components:
            value = component_values.get(component)

            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{trace_name}: representativeness component "
                    f"'{component}' must be numeric"
                )

            score_rows.append(
                {
                    "trace": trace_name,
                    "component": component,
                    "distance": float(value),
                }
            )

        summary_rows.append(
            {
                "trace": trace_name,
                "total_score": float(
                    representative["total_score"]
                ),
                "maximum_gpu_fraction_deviation": (
                    max(gpu_deltas)
                    if gpu_deltas
                    else 0.0
                ),
                "mean_runtime_cdf_deviation": (
                    sum(runtime_deltas)
                    / len(runtime_deltas)
                ),
                "maximum_runtime_cdf_deviation": (
                    max(runtime_deltas)
                ),
                "maximum_mapped_runtime_cdf_deviation": (
                    max(mapped_runtime_deltas)
                ),
                "two_gpu_job_count": int(
                    (
                        trace["mapped_num_gpus"] == 2
                    ).sum()
                ),
                "two_gpu_workload_count": int(
                    len(trace_two_gpu)
                ),
            }
        )

    if len(catalog_paths) != 1:
        raise ValueError(
            "All manifests must use the same workload catalog"
        )

    catalog_path = next(iter(catalog_paths))
    catalog = pd.read_csv(catalog_path)

    required_catalog_columns = {
        "workload_id",
        "num_gpus",
    }

    missing_catalog = (
        required_catalog_columns - set(catalog.columns)
    )

    if missing_catalog:
        raise ValueError(
            "Workload catalog is missing columns: "
            f"{sorted(missing_catalog)}"
        )

    catalog_two_gpu = sorted(
        catalog.loc[
            pd.to_numeric(
                catalog["num_gpus"],
                errors="raise",
            ).astype(int) == 2,
            "workload_id",
        ].astype(str)
    )

    coverage_rows = []

    for workload_id in catalog_two_gpu:
        traces = sorted(
            used_two_gpu.get(workload_id, set())
        )

        coverage_rows.append(
            {
                "workload_id": workload_id,
                "covered": bool(traces),
                "trace_count": len(traces),
                "traces": ",".join(traces),
            }
        )

    gpu_df = pd.DataFrame(gpu_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    interarrival_df = pd.DataFrame(
        interarrival_rows
    )
    score_df = pd.DataFrame(score_rows)
    summary_df = pd.DataFrame(summary_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    gpu_csv_path = (
        output_dir / "gpu_demand_fidelity.csv"
    )
    runtime_csv_path = (
        output_dir / "runtime_cdf_fidelity.csv"
    )
    interarrival_csv_path = (
        output_dir / "interarrival_fidelity.csv"
    )
    score_csv_path = (
        output_dir
        / "representativeness_scores.csv"
    )
    coverage_csv_path = (
        output_dir
        / "two_gpu_workload_coverage.csv"
    )

    gpu_df.to_csv(gpu_csv_path, index=False)
    runtime_df.to_csv(runtime_csv_path, index=False)
    interarrival_df.to_csv(
        interarrival_csv_path,
        index=False,
    )
    score_df.to_csv(score_csv_path, index=False)
    coverage_df.to_csv(
        coverage_csv_path,
        index=False,
    )

    gpu_figure_path = (
        output_dir / "gpu_demand_fidelity.png"
    )

    gpu_plot = gpu_df.copy()
    gpu_plot["category"] = (
        gpu_plot["trace"]
        + " "
        + gpu_plot["gpu_count"].astype(str)
        + " GPU"
    )

    x = list(range(len(gpu_plot)))
    width = 0.25

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        [value - width for value in x],
        gpu_plot["full_eligible"] * 100.0,
        width=width,
        label="Full eligible source",
    )
    axis.bar(
        x,
        gpu_plot["selected_window"] * 100.0,
        width=width,
        label="Selected window",
    )
    axis.bar(
        [value + width for value in x],
        gpu_plot["mapped_trace"] * 100.0,
        width=width,
        label="Mapped trace",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        gpu_plot["category"],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Jobs (%)")
    axis.set_title("GPU-demand fidelity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(gpu_figure_path, dpi=200)
    plt.close(figure)

    runtime_figure_path = (
        output_dir / "runtime_cdf_fidelity.png"
    )

    runtime_plot = runtime_df.copy()
    runtime_plot["category"] = (
        runtime_plot["trace"]
        + " "
        + runtime_plot["bucket_label"]
    )

    x = list(range(len(runtime_plot)))

    figure, axis = plt.subplots(figsize=(13, 5))
    axis.bar(
        [value - width for value in x],
        runtime_plot["full_eligible"] * 100.0,
        width=width,
        label="Full eligible source",
    )
    axis.bar(
        x,
        runtime_plot["selected_window"] * 100.0,
        width=width,
        label="Selected window",
    )
    axis.bar(
        [value + width for value in x],
        runtime_plot["mapped_trace"] * 100.0,
        width=width,
        label="Mapped trace",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        runtime_plot["category"],
        rotation=35,
        ha="right",
    )
    axis.set_ylabel("Cumulative jobs (%)")
    axis.set_title("Runtime-CDF fidelity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        runtime_figure_path,
        dpi=200,
    )
    plt.close(figure)

    interarrival_figure_path = (
        output_dir / "interarrival_fidelity.png"
    )

    interarrival_plot = interarrival_df.loc[
        interarrival_df["metric"].isin(
            [
                "interarrival_p50_s",
                "interarrival_p95_s",
            ]
        )
    ].copy()

    interarrival_plot["category"] = (
        interarrival_plot["trace"]
        + " "
        + interarrival_plot["metric"].map(
            {
                "interarrival_p50_s": "p50",
                "interarrival_p95_s": "p95",
            }
        )
    )

    x = list(range(len(interarrival_plot)))

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        [value - width / 2 for value in x],
        interarrival_plot["full_eligible"] / 60.0,
        width=width,
        label="Full eligible source",
    )
    axis.bar(
        [value + width / 2 for value in x],
        interarrival_plot["selected_window"] / 60.0,
        width=width,
        label="Selected/mapped trace",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        interarrival_plot["category"],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Interarrival time (minutes)")
    axis.set_title("Interarrival fidelity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        interarrival_figure_path,
        dpi=200,
    )
    plt.close(figure)

    score_figure_path = (
        output_dir
        / "representativeness_scores.png"
    )

    score_pivot = score_df.pivot(
        index="component",
        columns="trace",
        values="distance",
    )

    figure, axis = plt.subplots(figsize=(10, 5))

    score_pivot.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_xlabel("Score component")
    axis.set_ylabel("Distance")
    axis.set_title(
        "Representative-window score components"
    )
    axis.tick_params(
        axis="x",
        rotation=25,
    )
    figure.tight_layout()
    figure.savefig(
        score_figure_path,
        dpi=200,
    )
    plt.close(figure)

    report_path = (
        output_dir / "representative_fidelity.md"
    )

    lines = [
        "# Representative Trace Fidelity",
        "",
        "This report compares each full eligible production "
        "population, selected source window, and mapped "
        "executable trace.",
        "",
        "## Acceptance Summary",
        "",
        "| Trace | Score | Max GPU-demand deviation | "
        "Mean runtime-CDF deviation | "
        "Max runtime-CDF deviation | "
        "Max mapped-runtime deviation | "
        "2-GPU jobs | Distinct 2-GPU workloads |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_df.to_dict(
        orient="records"
    ):
        lines.append(
            f"| {row['trace']} | "
            f"{row['total_score']:.4f} | "
            f"{row['maximum_gpu_fraction_deviation']:.2%} | "
            f"{row['mean_runtime_cdf_deviation']:.2%} | "
            f"{row['maximum_runtime_cdf_deviation']:.2%} | "
            f"{row['maximum_mapped_runtime_cdf_deviation']:.2%} | "
            f"{row['two_gpu_job_count']} | "
            f"{row['two_gpu_workload_count']} |"
        )

    lines.extend(
        [
            "",
            "## GPU-Demand Fidelity",
            "",
            "![GPU-demand fidelity]"
            "(gpu_demand_fidelity.png)",
            "",
            "## Runtime-CDF Fidelity",
            "",
            "![Runtime-CDF fidelity]"
            "(runtime_cdf_fidelity.png)",
            "",
            "## Interarrival Fidelity",
            "",
            "![Interarrival fidelity]"
            "(interarrival_fidelity.png)",
            "",
            "## Representativeness Score",
            "",
            "![Representativeness score components]"
            "(representativeness_scores.png)",
            "",
            "## Suite-Level 2-GPU Workload Coverage",
            "",
            "| Workload | Covered | Traces |",
            "|---|---:|---|",
        ]
    )

    for row in coverage_df.to_dict(
        orient="records"
    ):
        lines.append(
            f"| {row['workload_id']} | "
            f"{'Yes' if row['covered'] else 'No'} | "
            f"{row['traces'] or '—'} |"
        )

    lines.extend(
        [
            "",
            "Selected-window fidelity is evaluated against "
            "the corresponding completed 1–2 GPU source "
            "population. Mapped-runtime differences are "
            "reported separately because the executable "
            "workload catalog may have a narrower runtime "
            "range than the production trace.",
            "",
        ]
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return {
        "fidelity_report_path": report_path,
        "gpu_fidelity_csv_path": gpu_csv_path,
        "runtime_fidelity_csv_path": runtime_csv_path,
        "interarrival_fidelity_csv_path": (
            interarrival_csv_path
        ),
        "score_csv_path": score_csv_path,
        "coverage_csv_path": coverage_csv_path,
        "gpu_fidelity_figure_path": (
            gpu_figure_path
        ),
        "runtime_fidelity_figure_path": (
            runtime_figure_path
        ),
        "interarrival_fidelity_figure_path": (
            interarrival_figure_path
        ),
        "score_figure_path": score_figure_path,
    }
