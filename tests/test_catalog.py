from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from trace_mapper.catalog import build_catalog, runtime_class


class CatalogTests(unittest.TestCase):
    def test_runtime_classes(self) -> None:
        self.assertEqual(runtime_class(1, 599.0), "light")
        self.assertEqual(runtime_class(1, 600.0), "medium_heavy")
        self.assertEqual(runtime_class(2, 100.0), "heavy_2gpu")

    def test_unsupported_gpu_count(self) -> None:
        with self.assertRaises(ValueError):
            runtime_class(3, 100.0)

    def test_build_catalog(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_root = root / "project"
            profiles_path = root / "profiles.csv"

            profiles = pd.DataFrame(
                [
                    {
                        "workload_id": "light-job",
                        "run_id": "run-1",
                        "spec_path": str(task_root / "specs/light.yaml"),
                        "gpu_count": 1,
                        "end_to_end_time_s": 500.0,
                        "exit_code": 0,
                        "gpu_memory_requirement_mib": 1000,
                    },
                    {
                        "workload_id": "medium-job",
                        "run_id": "run-2",
                        "spec_path": str(task_root / "specs/medium.yaml"),
                        "gpu_count": 1,
                        "end_to_end_time_s": 900.0,
                        "exit_code": 0,
                        "gpu_memory_requirement_mib": 2000,
                    },
                    {
                        "workload_id": "two-gpu-job",
                        "run_id": "run-3",
                        "spec_path": str(task_root / "specs/two_gpu.yaml"),
                        "gpu_count": 2,
                        "end_to_end_time_s": 1200.0,
                        "exit_code": 0,
                        "gpu_memory_requirement_mib": 3000,
                    },
                    {
                        "workload_id": "failed-job",
                        "run_id": "run-4",
                        "spec_path": str(task_root / "specs/failed.yaml"),
                        "gpu_count": 1,
                        "end_to_end_time_s": 100.0,
                        "exit_code": 1,
                        "gpu_memory_requirement_mib": 500,
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
                catalog["runtime_class"].value_counts().to_dict(),
                {
                    "light": 1,
                    "medium_heavy": 1,
                    "heavy_2gpu": 1,
                },
            )
            self.assertEqual(
                catalog.set_index("workload_id").loc[
                    "light-job", "task_path"
                ],
                "specs/light.yaml",
            )
            self.assertNotIn(
                "failed-job",
                catalog["workload_id"].tolist(),
            )


if __name__ == "__main__":
    unittest.main()