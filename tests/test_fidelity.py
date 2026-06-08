import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from trace_mapper.fidelity import (
    generate_representative_fidelity_report,
)


class RepresentativeFidelityTests(unittest.TestCase):
    def test_generates_fidelity_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            trace_path = root / "mapped.csv"
            catalog_path = root / "catalog.csv"
            manifest_path = root / "manifest.json"
            output_dir = root / "report"

            pd.DataFrame(
                [
                    {
                        "mapped_workload_id": "one-gpu",
                        "mapped_num_gpus": 1,
                        "mapped_solo_runtime_s": 300.0,
                        "source_interarrival_s": 0.0,
                    },
                    {
                        "mapped_workload_id": "two-gpu",
                        "mapped_num_gpus": 2,
                        "mapped_solo_runtime_s": 7200.0,
                        "source_interarrival_s": 20.0,
                    },
                ]
            ).to_csv(trace_path, index=False)

            pd.DataFrame(
                [
                    {
                        "workload_id": "one-gpu",
                        "num_gpus": 1,
                    },
                    {
                        "workload_id": "two-gpu",
                        "num_gpus": 2,
                    },
                ]
            ).to_csv(catalog_path, index=False)

            profile = {
                "job_count": 2,
                "gpu_demand_fractions": {
                    "1": 0.5,
                    "2": 0.5,
                },
                "runtime_cdf": {
                    "under_10m": 0.5,
                    "under_1h": 0.5,
                    "under_6h": 1.0,
                    "under_1d": 1.0,
                },
                "interarrival_p50_s": 10.0,
                "interarrival_p95_s": 19.0,
                "zero_interarrival_fraction": 0.5,
                "gpu_service_fractions": {
                    "1": 0.02,
                    "2": 0.98,
                },
            }

            manifest = {
                "configuration": {
                    "source_cluster": "philly",
                    "supported_gpu_counts": [1, 2],
                },
                "selection": {
                    "representativeness": {
                        "total_score": 0.1,
                        "components": {
                            "gpu_demand_distance": 0.0,
                            "runtime_cdf_distance": 0.0,
                            "interarrival_distance": 0.1,
                            "burst_fraction_distance": 0.0,
                            "gpu_service_distance": 0.0,
                        },
                        "target_profile": profile,
                        "selected_profile": profile,
                    }
                },
                "inputs": {
                    "workload_catalog_path": str(
                        catalog_path
                    ),
                },
                "output": {
                    "trace_path": str(trace_path),
                },
            }

            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            outputs = (
                generate_representative_fidelity_report(
                    [manifest_path],
                    output_dir=output_dir,
                )
            )

            for path in outputs.values():
                self.assertTrue(path.is_file())

            report = outputs[
                "fidelity_report_path"
            ].read_text(encoding="utf-8")

            self.assertIn(
                "Representative Trace Fidelity",
                report,
            )
            self.assertIn("Philly", report)
            self.assertIn("two-gpu", report)

            coverage = pd.read_csv(
                outputs["coverage_csv_path"]
            )

            self.assertEqual(len(coverage), 1)
            self.assertTrue(
                bool(coverage.iloc[0]["covered"])
            )


if __name__ == "__main__":
    unittest.main()
