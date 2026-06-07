from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from trace_mapper.catalog import build_catalog


class CatalogTests(unittest.TestCase):
    def test_build_catalog_memory_semantics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_root = root / "project"
            profiles_path = root / "profiles.csv"

            profiles = pd.DataFrame(
                [
                    {
                        "workload_id": "single-gpu-job",
                        "run_id": "run-1",
                        "spec_path": str(
                            task_root / "specs/single_gpu.yaml"
                        ),
                        "gpu_count": 1,
                        "end_to_end_time_s": 500.0,
                        "exit_code": 0,
                        "gpu_memory_requirement_mib": 1000.0,
                        "gpu_memory_peak_full_mib": 1200.0,
                    },
                    {
                        "workload_id": "two-gpu-job",
                        "run_id": "run-2",
                        "spec_path": str(
                            task_root / "specs/two_gpu.yaml"
                        ),
                        "gpu_count": 2,
                        "end_to_end_time_s": 1200.0,
                        "exit_code": 0,
                        "gpu_memory_requirement_mib": 4800.0,
                        "gpu_memory_peak_full_mib_gpu_a": 3000.0,
                        "gpu_memory_peak_full_mib_gpu_b": 2200.0,
                        "gpu_memory_peak_full_mib_sum": 5200.0,
                    },
                    {
                        # Represents the Llama case:
                        # no declared requirement, but measured peak exists.
                        "workload_id": "llama-like-job",
                        "run_id": "run-3",
                        "spec_path": str(
                            task_root / "specs/llama.yaml"
                        ),
                        "gpu_count": 1,
                        "end_to_end_time_s": 900.0,
                        "exit_code": 0,
                        "gpu_memory_requirement_mib": None,
                        "gpu_memory_peak_full_mib": 28712.0,
                    },
                    {
                        "workload_id": "failed-job",
                        "run_id": "run-4",
                        "spec_path": str(
                            task_root / "specs/failed.yaml"
                        ),
                        "gpu_count": 1,
                        "end_to_end_time_s": 100.0,
                        "exit_code": 1,
                        "gpu_memory_requirement_mib": 500.0,
                        "gpu_memory_peak_full_mib": 600.0,
                    },
                ]
            )
            profiles.to_csv(profiles_path, index=False)

            catalog = build_catalog(
                [profiles_path],
                task_root=task_root,
            )

            self.assertEqual(len(catalog), 3)

            self.assertEqual(
                list(catalog.columns),
                [
                    "workload_id",
                    "task_path",
                    "num_gpus",
                    "solo_runtime_s",
                    "declared_memory_requirement_mib",
                    "measured_peak_memory_per_gpu_mib",
                    "measured_peak_memory_total_mib",
                    "source_run_id",
                ],
            )

            indexed = catalog.set_index("workload_id")

            single = indexed.loc["single-gpu-job"]
            self.assertEqual(single["task_path"], "specs/single_gpu.yaml")
            self.assertEqual(single["num_gpus"], 1)
            self.assertEqual(single["solo_runtime_s"], 500.0)
            self.assertEqual(
                single["declared_memory_requirement_mib"],
                1000.0,
            )
            self.assertEqual(
                single["measured_peak_memory_per_gpu_mib"],
                1200.0,
            )
            self.assertEqual(
                single["measured_peak_memory_total_mib"],
                1200.0,
            )

            two_gpu = indexed.loc["two-gpu-job"]
            self.assertEqual(two_gpu["num_gpus"], 2)
            self.assertEqual(
                two_gpu["declared_memory_requirement_mib"],
                4800.0,
            )
            self.assertEqual(
                two_gpu["measured_peak_memory_per_gpu_mib"],
                3000.0,
            )
            self.assertEqual(
                two_gpu["measured_peak_memory_total_mib"],
                5200.0,
            )

            llama = indexed.loc["llama-like-job"]
            self.assertTrue(
                pd.isna(llama["declared_memory_requirement_mib"])
            )
            self.assertEqual(
                llama["measured_peak_memory_per_gpu_mib"],
                28712.0,
            )
            self.assertEqual(
                llama["measured_peak_memory_total_mib"],
                28712.0,
            )

            self.assertNotIn(
                "failed-job",
                catalog["workload_id"].tolist(),
            )


if __name__ == "__main__":
    unittest.main()