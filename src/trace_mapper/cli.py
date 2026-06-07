from __future__ import annotations

import argparse

from pathlib import Path

from trace_mapper.catalog import build_catalog

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

    return parser


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

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())