from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-mapper",
        description="Generate reproducible DL workload traces from production traces.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "build-catalog",
        help="Build a workload catalog from AEGIS solo-profiling results.",
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
        print("build-catalog: not implemented yet")
        return 0

    if args.command == "generate":
        print("generate: not implemented yet")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())