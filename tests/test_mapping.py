import unittest

import pandas as pd

from trace_mapper.mapping import map_jobs_to_workloads


class WorkloadMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_jobs = pd.DataFrame(
            [
                {
                    "source_trace": "helios",
                    "source_cluster": "venus",
                    "source_job_id": "job-1",
                    "source_duration_s": 100.0,
                    "source_num_gpus": 1,
                    "submit_time_s": 0.0,
                    "source_interarrival_s": 0.0,
                },
                {
                    "source_trace": "helios",
                    "source_cluster": "venus",
                    "source_job_id": "job-2",
                    "source_duration_s": 200.0,
                    "source_num_gpus": 1,
                    "submit_time_s": 10.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    "source_trace": "helios",
                    "source_cluster": "venus",
                    "source_job_id": "job-3",
                    "source_duration_s": 300.0,
                    "source_num_gpus": 1,
                    "submit_time_s": 20.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    "source_trace": "helios",
                    "source_cluster": "venus",
                    "source_job_id": "job-4",
                    "source_duration_s": 1000.0,
                    "source_num_gpus": 2,
                    "submit_time_s": 30.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    # Unsupported on the configured server.
                    "source_trace": "helios",
                    "source_cluster": "venus",
                    "source_job_id": "job-5",
                    "source_duration_s": 2000.0,
                    "source_num_gpus": 8,
                    "submit_time_s": 40.0,
                    "source_interarrival_s": 10.0,
                },
            ]
        )

        self.catalog = pd.DataFrame(
            [
                {
                    "workload_id": "short-1gpu",
                    "task_path": "specs/short.yaml",
                    "num_gpus": 1,
                    "solo_runtime_s": 50.0,
                },
                {
                    "workload_id": "middle-1gpu",
                    "task_path": "specs/middle.yaml",
                    "num_gpus": 1,
                    "solo_runtime_s": 150.0,
                },
                {
                    "workload_id": "long-1gpu",
                    "task_path": "specs/long.yaml",
                    "num_gpus": 1,
                    "solo_runtime_s": 500.0,
                },
                {
                    "workload_id": "two-gpu",
                    "task_path": "specs/two_gpu.yaml",
                    "num_gpus": 2,
                    "solo_runtime_s": 800.0,
                },
            ]
        )

    def test_matches_gpu_count_and_runtime_quantile(self) -> None:
        mapped = map_jobs_to_workloads(
            self.source_jobs,
            self.catalog,
            supported_gpu_counts=(1, 2),
        )

        self.assertEqual(len(mapped), 4)
        self.assertNotIn(
            "job-5",
            mapped["source_job_id"].tolist(),
        )

        indexed = mapped.set_index("source_job_id")

        self.assertEqual(
            indexed.loc["job-1", "mapped_workload_id"],
            "short-1gpu",
        )
        self.assertEqual(
            indexed.loc["job-2", "mapped_workload_id"],
            "middle-1gpu",
        )
        self.assertEqual(
            indexed.loc["job-3", "mapped_workload_id"],
            "long-1gpu",
        )
        self.assertEqual(
            indexed.loc["job-4", "mapped_workload_id"],
            "two-gpu",
        )

        self.assertTrue(
            (
                mapped["source_num_gpus"]
                == mapped["mapped_num_gpus"]
            ).all()
        )

        self.assertTrue(
            (
                mapped["runtime_quantile_distance"] >= 0
            ).all()
        )

    def test_rejects_missing_catalog_gpu_count(self) -> None:
        catalog_without_two_gpu = self.catalog.loc[
            self.catalog["num_gpus"] == 1
        ]

        with self.assertRaisesRegex(
            ValueError,
            "no entries for GPU counts",
        ):
            map_jobs_to_workloads(
                self.source_jobs,
                catalog_without_two_gpu,
                supported_gpu_counts=(1, 2),
            )


if __name__ == "__main__":
    unittest.main()
