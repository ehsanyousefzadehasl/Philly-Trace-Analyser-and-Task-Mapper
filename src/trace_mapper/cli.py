from __future__ import annotations

import argparse

from pathlib import Path

from trace_mapper.catalog import build_catalog
import json
from pathlib import Path

from trace_mapper.sources.helios import load_helios_jobs
from trace_mapper.sources.philly import load_philly_jobs
from trace_mapper.summary import (
    gpu_demand_distribution,
    summarize_source_jobs,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-mapper",
        description="Generate reproducible DL workload traces from production traces.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser(
        "build-catalog",
        help="Build a workload catalog from solo profiling results.",
    )
    catalog_parser.add_argument(
        "--profiles",
        type=Path,
        nargs="+",
        required=True,
        help="One or more solo-profile CSV files.",
    )
    catalog_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output workload-catalog CSV.",
    )
    catalog_parser.add_argument(
        "--task-root",
        type=Path,
        default=None,
        help="Optional root used to convert task paths to portable relative paths.",
    )

    subparsers.add_parser(
        "generate",
        help="Map production-trace arrivals to profiled workloads.",
    )

    summary_parser = subparsers.add_parser(
        "summarize-source",
        help="Characterize a normalized production trace.",
    )

    summary_parser.add_argument(
        "--source-format",
        choices=["philly", "helios"],
        required=True,
        help="Input production-trace format.",
    )

    summary_parser.add_argument(
        "--trace-path",
        type=Path,
        required=True,
        help="Path to the raw production trace.",
    )

    summary_parser.add_argument(
        "--cluster-name",
        default=None,
        help="Cluster name, required for Helios traces.",
    )

    summary_parser.add_argument(
        "--supported-gpu-counts",
        type=int,
        nargs="+",
        default=[1, 2],
        help="GPU demands supported by the target testbed.",
    )

    summary_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated summary artifacts.",
    )

    return parser

def load_source_jobs(
    *,
    source_format: str,
    trace_path: Path,
    cluster_name: str | None,
):
    if source_format == "philly":
        return load_philly_jobs(trace_path)

    if source_format == "helios":
        if cluster_name is None or not cluster_name.strip():
            raise ValueError(
                "--cluster-name is required for Helios traces"
            )

        return load_helios_jobs(
            trace_path,
            cluster_name=cluster_name,
        )

    raise ValueError(
        f"Unsupported source format: {source_format}"
    )


def write_source_summary(
    *,
    jobs,
    supported_gpu_counts,
    output_dir: Path,
) -> tuple[Path, Path]:
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

    distribution.to_csv(
        distribution_path,
        index=False,
    )

    return summary_path, distribution_path

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-catalog":
        catalog = build_catalog(
            args.profiles,
            task_root=args.task_root,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(args.output, index=False)

        print(f"Wrote {len(catalog)} workloads to {args.output}")

        print("\nGPU-count distribution:")
        print(
            catalog["num_gpus"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "\nMissing declared-memory values:",
            int(catalog["declared_memory_requirement_mib"].isna().sum()),
        )
        print(
            "Missing measured peak-memory values:",
            int(catalog["measured_peak_memory_per_gpu_mib"].isna().sum()),
        )

        return 0

    if args.command == "generate":
        print("generate: not implemented yet")
        return 0

    if args.command == "summarize-source":
        jobs = load_source_jobs(
            source_format=args.source_format,
            trace_path=args.trace_path,
            cluster_name=args.cluster_name,
        )

        summary_path, distribution_path = write_source_summary(
            jobs=jobs,
            supported_gpu_counts=args.supported_gpu_counts,
            output_dir=args.output_dir,
        )

        summary = summarize_source_jobs(
            jobs,
            supported_gpu_counts=args.supported_gpu_counts,
        )

        print(
            f"Normalized {summary['valid_job_count']} jobs from "
            f"{summary['source_cluster']}"
        )
        print(
            "Supported jobs:",
            summary["supported_job_count"],
            f"({summary['supported_job_fraction']:.2%})",
        )
        print(f"Wrote summary to {summary_path}")
        print(
            "Wrote GPU-demand distribution to "
            f"{distribution_path}"
        )

        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())