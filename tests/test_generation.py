import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from trace_mapper.generation import run_generation


class TraceGenerationTests(unittest.TestCase):
    def test_generates_trace_and_manifest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            source_path = root / "cluster_job_log"
            catalog_path = root / "catalog.csv"
            trace_path = root / "mapped.csv"
            manifest_path = root / "mapped.manifest.json"
            config_path = root / "generation.yaml"

            source_jobs = [
                {
                    "status": "Pass",
                    "jobid": "job-1",
                    "submitted_time": "2017-01-01 00:00:00",
                    "attempts": [
                        {
                            "start_time": "2017-01-01 00:00:05",
                            "end_time": "2017-01-01 00:01:05",
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
                            "end_time": "2017-01-01 00:02:20",
                            "detail": [
                                {
                                    "ip": "node-2",
                                    "gpus": ["gpu0"],
                                }
                            ],
                        }
                    ],
                },
                {
                    "status": "Pass",
                    "jobid": "job-3",
                    "submitted_time": "2017-01-01 00:00:20",
                    "attempts": [
                        {
                            "start_time": "2017-01-01 00:00:30",
                            "end_time": "2017-01-01 00:03:30",
                            "detail": [
                                {
                                    "ip": "node-3",
                                    "gpus": ["gpu0", "gpu1"],
                                }
                            ],
                        }
                    ],
                },
            ]

            source_path.write_text(
                json.dumps(source_jobs),
                encoding="utf-8",
            )

            catalog = pd.DataFrame(
                [
                    {
                        "workload_id": "short-1gpu",
                        "task_path": "specs/short.yaml",
                        "num_gpus": 1,
                        "solo_runtime_s": 50.0,
                    },
                    {
                        "workload_id": "long-1gpu",
                        "task_path": "specs/long.yaml",
                        "num_gpus": 1,
                        "solo_runtime_s": 150.0,
                    },
                    {
                        "workload_id": "two-gpu",
                        "task_path": "specs/two_gpu.yaml",
                        "num_gpus": 2,
                        "solo_runtime_s": 200.0,
                    },
                ]
            )
            catalog.to_csv(catalog_path, index=False)

            config_path.write_text(
                f"""
version: 1

source:
  format: philly
  trace_path: {source_path}
  cluster_name: philly

mapping:
  workload_catalog_path: {catalog_path}
  seed: 42
  num_jobs: 3
  supported_gpu_counts: [1, 2]
  gpu_demand_matching: exact
  duration_matching: nearest_quantile
  unsupported_gpu_policy: filter
  preserve_arrivals: true

output:
  trace_path: {trace_path}
  manifest_path: {manifest_path}
""".strip(),
                encoding="utf-8",
            )

            outputs = run_generation(config_path)

            self.assertEqual(
                outputs["mapped_job_count"],
                3,
            )
            self.assertTrue(trace_path.is_file())
            self.assertTrue(manifest_path.is_file())

            mapped = pd.read_csv(trace_path)

            self.assertEqual(len(mapped), 3)
            self.assertEqual(
                mapped.columns[:2].tolist(),
                ["submit_time_s", "task_path"],
            )
            self.assertTrue(
                mapped["submit_time_s"].is_monotonic_increasing
            )
            self.assertTrue(
                (
                    mapped["source_num_gpus"]
                    == mapped["mapped_num_gpus"]
                ).all()
            )

            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                manifest["mapping"]["mapped_job_count"],
                3,
            )
            self.assertEqual(
                manifest["configuration"][
                    "supported_gpu_counts"
                ],
                [1, 2],
            )
            self.assertEqual(
                len(
                    manifest["inputs"][
                        "source_trace_sha256"
                    ]
                ),
                64,
            )
            self.assertEqual(
                len(
                    manifest["output"][
                        "trace_sha256"
                    ]
                ),
                64,
            )


if __name__ == "__main__":
    unittest.main()
