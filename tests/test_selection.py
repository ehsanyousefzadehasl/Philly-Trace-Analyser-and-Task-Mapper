import unittest

import pandas as pd

from trace_mapper.selection import select_contiguous_job_window


class SourceWindowSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = pd.DataFrame(
            [
                {
                    "source_job_id": "job-1",
                    "source_num_gpus": 1,
                    "submit_time_s": 0.0,
                    "source_interarrival_s": 0.0,
                },
                {
                    "source_job_id": "job-2",
                    "source_num_gpus": 2,
                    "submit_time_s": 10.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    # Unsupported job. Its elapsed time must not vanish.
                    "source_job_id": "job-3",
                    "source_num_gpus": 8,
                    "submit_time_s": 20.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    "source_job_id": "job-4",
                    "source_num_gpus": 1,
                    "submit_time_s": 30.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    "source_job_id": "job-5",
                    "source_num_gpus": 2,
                    "submit_time_s": 40.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    "source_job_id": "job-6",
                    "source_num_gpus": 1,
                    "submit_time_s": 50.0,
                    "source_interarrival_s": 10.0,
                },
            ]
        )

    def test_selects_reproducible_contiguous_window(self) -> None:
        window, metadata = select_contiguous_job_window(
            self.jobs,
            num_jobs=3,
            supported_gpu_counts=(1, 2),
            seed=7,
        )

        # Eligible sequence:
        # job-1, job-2, job-4, job-5, job-6
        # Seed 7 selects eligible positions 1:4.
        self.assertEqual(
            window["source_job_id"].tolist(),
            ["job-2", "job-4", "job-5"],
        )

        self.assertEqual(
            window["source_trace_submit_time_s"].tolist(),
            [10.0, 30.0, 40.0],
        )

        # Submission times are normalized to the selected window.
        self.assertEqual(
            window["submit_time_s"].tolist(),
            [0.0, 20.0, 30.0],
        )

        # The unsupported job occurred during the 20-second gap.
        self.assertEqual(
            window["source_interarrival_s"].tolist(),
            [0.0, 20.0, 10.0],
        )

        self.assertEqual(
            window["source_window_job_index"].tolist(),
            [0, 1, 2],
        )

        self.assertEqual(
            metadata["selection_method"],
            "random_contiguous_eligible_jobs",
        )
        self.assertEqual(metadata["source_job_count"], 6)
        self.assertEqual(metadata["eligible_job_count"], 5)
        self.assertEqual(metadata["filtered_job_count"], 1)
        self.assertEqual(metadata["eligible_start_index"], 1)
        self.assertEqual(metadata["selected_arrival_span_s"], 30.0)

    def test_same_seed_produces_same_window(self) -> None:
        first, first_metadata = select_contiguous_job_window(
            self.jobs,
            num_jobs=3,
            supported_gpu_counts=(1, 2),
            seed=42,
        )

        second, second_metadata = select_contiguous_job_window(
            self.jobs,
            num_jobs=3,
            supported_gpu_counts=(1, 2),
            seed=42,
        )

        self.assertEqual(
            first["source_job_id"].tolist(),
            second["source_job_id"].tolist(),
        )
        self.assertEqual(first_metadata, second_metadata)

    def test_rejects_too_many_requested_jobs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "only 5 eligible jobs",
        ):
            select_contiguous_job_window(
                self.jobs,
                num_jobs=6,
                supported_gpu_counts=(1, 2),
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
