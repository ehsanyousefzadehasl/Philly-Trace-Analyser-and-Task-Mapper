import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from trace_mapper.report import generate_trace_report


class TraceReportTests(unittest.TestCase):
    def test_generates_markdown_and_figures(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            trace_path = root / "mapped.csv"
            jobs_path = root / "exclusive_jobs.csv"
            manifest_path = root / "mapped.manifest.json"
            output_dir = root / "report"

            pd.DataFrame(
                [
                    {
                        "mapped_workload_id": "job-a",
                        "mapped_num_gpus": 1,
                        "runtime_quantile_distance": 0.1,
                    },
                    {
                        "mapped_workload_id": "job-b",
                        "mapped_num_gpus": 2,
                        "runtime_quantile_distance": 0.2,
                    },
                ]
            ).to_csv(trace_path, index=False)

            pd.DataFrame(
                [
                    {
                        "exclusive_waiting_time_s": 0.0,
                        "mapped_solo_runtime_s": 100.0,
                    },
                    {
                        "exclusive_waiting_time_s": 20.0,
                        "mapped_solo_runtime_s": 200.0,
                    },
                ]
            ).to_csv(jobs_path, index=False)

            manifest = {
                "configuration": {
                    "source_cluster": "philly",
                    "seed": 42,
                    "supported_gpu_counts": [1, 2],
                },
                "selection": {
                    "selected_arrival_span_s": 10.0,
                },
                "mapping": {
                    "unique_workload_count": 2,
                    "runtime_quantile_distance_mean": 0.15,
                    "runtime_quantile_distance_p95": 0.195,
                    "runtime_quantile_distance_max": 0.2,
                },
                "exclusive_simulation": {
                    "server_gpu_count": 3,
                    "makespan_s": 220.0,
                    "waiting_mean_s": 10.0,
                    "waiting_p95_s": 19.0,
                    "waiting_p99_s": 19.8,
                    "jct_mean_s": 160.0,
                    "jct_p95_s": 214.0,
                    "jct_p99_s": 218.8,
                },
                "output": {
                    "trace_path": str(trace_path),
                    "exclusive_job_metrics_path": str(
                        jobs_path
                    ),
                },
            }

            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            outputs = generate_trace_report(
                manifest_path,
                output_dir=output_dir,
            )

            self.assertTrue(
                outputs["report_path"].is_file()
            )
            self.assertTrue(
                outputs["gpu_demand_path"].is_file()
            )
            self.assertTrue(
                outputs["timeline_path"].is_file()
            )

            report = outputs["report_path"].read_text(
                encoding="utf-8"
            )

            self.assertIn("Generated Trace Report", report)
            self.assertIn("Philly", report)
            self.assertIn("220.00 s", report)


if __name__ == "__main__":
    unittest.main()
