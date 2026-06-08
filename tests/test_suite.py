import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from trace_mapper.suite import run_summary_suite


class SummarySuiteTests(unittest.TestCase):
    def test_generates_multi_trace_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            philly_path = root / "philly.json"
            helios_path = root / "saturn.csv"
            config_path = root / "suite.yaml"

            philly_jobs = [
                {
                    "status": "Pass",
                    "jobid": "philly-1",
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
                }
            ]

            philly_path.write_text(
                json.dumps(philly_jobs),
                encoding="utf-8",
            )

            helios_jobs = [
                {
                    "job_id": 1,
                    "gpu_num": 2,
                    "node_num": 1,
                    "state": "COMPLETED",
                    "submit_time": "2020-01-01 00:00:00",
                    "start_time": "2020-01-01 00:00:10",
                    "end_time": "2020-01-01 00:01:10",
                    "duration": 60,
                    "queue": 10,
                }
            ]

            pd.DataFrame(helios_jobs).to_csv(
                helios_path,
                index=False,
            )

            config_path.write_text(
                f"""
version: 1
root_dir: .

supported_gpu_counts: [1, 2]

sources:
  - name: philly
    format: philly
    trace_path: {philly_path}

  - name: saturn
    format: helios
    cluster_name: saturn
    trace_path: {helios_path}

output:
  artifact_dir: {root / "outputs"}
  markdown_path: {root / "report.md"}
""".strip(),
                encoding="utf-8",
            )

            outputs = run_summary_suite(config_path)

            self.assertTrue(
                outputs["suite_csv_path"].is_file()
            )
            self.assertTrue(
                outputs["suite_json_path"].is_file()
            )
            self.assertTrue(
                outputs["markdown_path"].is_file()
            )

            suite = pd.read_csv(
                outputs["suite_csv_path"]
            )

            self.assertTrue(
                outputs["coverage_path"].is_file()
            )
            self.assertTrue(
                outputs["gpu_mix_path"].is_file()
            )
            self.assertTrue(
                outputs["runtime_path"].is_file()
            )
            self.assertTrue(
                outputs["queue_path"].is_file()
            )

            self.assertEqual(len(suite), 2)
            self.assertEqual(
                set(suite["source_cluster"]),
                {"philly", "saturn"},
            )

            report = outputs["markdown_path"].read_text(
                encoding="utf-8"
            )

            self.assertIn("Philly", report)
            self.assertIn("Saturn", report)

            
            self.assertIn(
                "trace_characterization/supported_coverage.png",
                report,
            )
            self.assertIn(
                "trace_characterization/gpu_demand_mix.png",
                report,
            )

if __name__ == "__main__":
    unittest.main()
