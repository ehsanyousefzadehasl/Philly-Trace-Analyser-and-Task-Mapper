import unittest

from trace_mapper.cli import build_parser

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from trace_mapper.cli import build_parser, main

class CliTests(unittest.TestCase):
    def test_build_catalog_command_parses(self) -> None:
        args = build_parser().parse_args(
            [
                "build-catalog",
                "--profiles",
                "profiles_1gpu.csv",
                "profiles_2gpu.csv",
                "--output",
                "workload_catalog.csv",
                "--task-root",
                "/workloads",
            ]
        )

        self.assertEqual(args.command, "build-catalog")
        self.assertEqual(
            [str(path) for path in args.profiles],
            ["profiles_1gpu.csv", "profiles_2gpu.csv"],
        )
        self.assertEqual(str(args.output), "workload_catalog.csv")
        self.assertEqual(str(args.task_root), "/workloads")

    def test_generate_command_parses(self) -> None:
        args = build_parser().parse_args(["generate"])
        self.assertEqual(args.command, "generate")

def test_summarize_source_command_parses(self) -> None:
    args = build_parser().parse_args(
        [
            "summarize-source",
            "--source-format",
            "helios",
            "--trace-path",
            "cluster_log.csv",
            "--cluster-name",
            "venus",
            "--supported-gpu-counts",
            "1",
            "2",
            "--output-dir",
            "outputs/venus",
        ]
    )

    self.assertEqual(args.command, "summarize-source")
    self.assertEqual(args.source_format, "helios")
    self.assertEqual(args.cluster_name, "venus")
    self.assertEqual(args.supported_gpu_counts, [1, 2])


def test_summarize_philly_writes_artifacts(self) -> None:
    jobs = [
        {
            "status": "Pass",
            "jobid": "job-1",
            "submitted_time": "2017-01-01 00:00:00",
            "attempts": [
                {
                    "start_time": "2017-01-01 00:00:05",
                    "end_time": "2017-01-01 00:00:35",
                    "detail": [
                        {
                            "ip": "node-1",
                            "gpus": ["gpu0"],
                        }
                    ],
                }
            ],
        },
        {
            "status": "Pass",
            "jobid": "job-2",
            "submitted_time": "2017-01-01 00:00:10",
            "attempts": [
                {
                    "start_time": "2017-01-01 00:00:20",
                    "end_time": "2017-01-01 00:01:20",
                    "detail": [
                        {
                            "ip": "node-2",
                            "gpus": ["gpu0", "gpu1"],
                        }
                    ],
                }
            ],
        },
    ]

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        trace_path = root / "cluster_job_log"
        output_dir = root / "analysis"

        trace_path.write_text(
            json.dumps(jobs),
            encoding="utf-8",
        )

        argv = [
            "trace-mapper",
            "summarize-source",
            "--source-format",
            "philly",
            "--trace-path",
            str(trace_path),
            "--supported-gpu-counts",
            "1",
            "2",
            "--output-dir",
            str(output_dir),
        ]

        with patch("sys.argv", argv):
            return_code = main()

        self.assertEqual(return_code, 0)

        summary_path = output_dir / "source_summary.json"
        distribution_path = (
            output_dir / "gpu_demand_distribution.csv"
        )

        self.assertTrue(summary_path.is_file())
        self.assertTrue(distribution_path.is_file())

        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        self.assertEqual(summary["valid_job_count"], 2)
        self.assertEqual(summary["supported_job_count"], 2)
        self.assertEqual(
            summary["supported_job_fraction"],
            1.0,
        )

if __name__ == "__main__":
    unittest.main()