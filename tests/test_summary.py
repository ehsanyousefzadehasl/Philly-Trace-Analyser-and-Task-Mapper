import unittest

import pandas as pd

from trace_mapper.summary import (
    gpu_demand_distribution,
    summarize_source_jobs,
)


class SourceSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = pd.DataFrame(
            [
                {
                    "source_trace": "helios",
                    "source_cluster": "saturn",
                    "source_job_id": "job-1",
                    "source_duration_s": 100.0,
                    "source_queue_wait_s": 10.0,
                    "source_num_gpus": 1,
                    "source_multi_node": False,
                    "submit_time_s": 0.0,
                },
                {
                    "source_trace": "helios",
                    "source_cluster": "saturn",
                    "source_job_id": "job-2",
                    "source_duration_s": 200.0,
                    "source_queue_wait_s": 20.0,
                    "source_num_gpus": 2,
                    "source_multi_node": False,
                    "submit_time_s": 50.0,
                },
                {
                    "source_trace": "helios",
                    "source_cluster": "saturn",
                    "source_job_id": "job-3",
                    "source_duration_s": 300.0,
                    "source_queue_wait_s": 30.0,
                    "source_num_gpus": 4,
                    "source_multi_node": True,
                    "submit_time_s": 100.0,
                },
            ]
        )

    def test_summarizes_supported_server_scope(self) -> None:
        summary = summarize_source_jobs(
            self.jobs,
            supported_gpu_counts=(1, 2),
        )

        self.assertEqual(summary["source_trace"], "helios")
        self.assertEqual(summary["source_cluster"], "saturn")

        self.assertEqual(summary["valid_job_count"], 3)
        self.assertEqual(summary["supported_job_count"], 2)
        self.assertEqual(summary["unsupported_job_count"], 1)

        self.assertAlmostEqual(
            summary["supported_job_fraction"],
            2 / 3,
        )

        self.assertEqual(summary["single_gpu_job_count"], 1)
        self.assertEqual(summary["two_gpu_job_count"], 1)
        self.assertEqual(summary["multi_node_job_count"], 1)

        self.assertEqual(summary["arrival_span_s"], 100.0)

        # GPU service time:
        # 1*100 + 2*200 + 4*300 = 1700
        self.assertEqual(
            summary["total_gpu_service_time_s"],
            1700.0,
        )

        # Supported jobs:
        # 1*100 + 2*200 = 500
        self.assertEqual(
            summary["supported_gpu_service_time_s"],
            500.0,
        )
        self.assertAlmostEqual(
            summary["supported_gpu_service_time_fraction"],
            500 / 1700,
        )

        self.assertEqual(
            summary["all_duration_mean_s"],
            200.0,
        )
        self.assertEqual(
            summary["supported_duration_mean_s"],
            150.0,
        )
        self.assertEqual(
            summary["supported_queue_wait_mean_s"],
            15.0,
        )

    def test_builds_gpu_demand_distribution(self) -> None:
        distribution = gpu_demand_distribution(self.jobs)

        self.assertEqual(
            distribution["source_num_gpus"].tolist(),
            [1, 2, 4],
        )
        self.assertEqual(
            distribution["job_count"].tolist(),
            [1, 1, 1],
        )

        for fraction in distribution["job_fraction"]:
            self.assertAlmostEqual(fraction, 1 / 3)

    def test_rejects_missing_columns(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "missing required columns",
        ):
            summarize_source_jobs(
                self.jobs.drop(columns=["source_duration_s"])
            )


if __name__ == "__main__":
    unittest.main()
