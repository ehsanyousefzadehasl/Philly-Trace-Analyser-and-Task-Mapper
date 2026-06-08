import unittest

import pandas as pd

from trace_mapper.representative import (
    build_representativeness_profile,
    score_representativeness,
    select_representative_job_window,
)


class RepresentativeProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = pd.DataFrame(
            [
                {
                    "source_duration_s": 300.0,
                    "source_interarrival_s": 0.0,
                    "source_num_gpus": 1,
                },
                {
                    "source_duration_s": 900.0,
                    "source_interarrival_s": 10.0,
                    "source_num_gpus": 1,
                },
                {
                    "source_duration_s": 7200.0,
                    "source_interarrival_s": 20.0,
                    "source_num_gpus": 2,
                },
                {
                    "source_duration_s": 30000.0,
                    "source_interarrival_s": 30.0,
                    "source_num_gpus": 2,
                },
                {
                    # Unsupported and excluded from the profile.
                    "source_duration_s": 100.0,
                    "source_interarrival_s": 40.0,
                    "source_num_gpus": 8,
                },
            ]
        )

    def test_builds_profile_for_supported_jobs(self) -> None:
        profile = build_representativeness_profile(
            self.jobs,
            supported_gpu_counts=(1, 2),
        )

        self.assertEqual(profile["job_count"], 4)

        self.assertEqual(
            profile["gpu_demand_fractions"],
            {
                "1": 0.5,
                "2": 0.5,
            },
        )

        self.assertEqual(
            profile["runtime_cdf"]["under_10m"],
            0.25,
        )
        self.assertEqual(
            profile["runtime_cdf"]["under_1h"],
            0.5,
        )
        self.assertEqual(
            profile["runtime_cdf"]["under_6h"],
            0.75,
        )
        self.assertEqual(
            profile["runtime_cdf"]["under_1d"],
            1.0,
        )

        self.assertEqual(
            profile["zero_interarrival_fraction"],
            0.25,
        )

        self.assertAlmostEqual(
            sum(
                profile[
                    "gpu_service_fractions"
                ].values()
            ),
            1.0,
        )

    def test_identical_profiles_have_zero_score(self) -> None:
        profile = build_representativeness_profile(
            self.jobs,
            supported_gpu_counts=(1, 2),
        )

        score = score_representativeness(
            profile,
            profile,
        )

        self.assertAlmostEqual(
            score["total_score"],
            0.0,
        )

        for value in score["components"].values():
            self.assertAlmostEqual(value, 0.0)

    def test_distorted_profile_has_positive_score(self) -> None:
        target = build_representativeness_profile(
            self.jobs,
            supported_gpu_counts=(1, 2),
        )

        candidate_jobs = pd.DataFrame(
            [
                {
                    "source_duration_s": 100.0,
                    "source_interarrival_s": 0.0,
                    "source_num_gpus": 1,
                },
                {
                    "source_duration_s": 120.0,
                    "source_interarrival_s": 0.0,
                    "source_num_gpus": 1,
                },
                {
                    "source_duration_s": 140.0,
                    "source_interarrival_s": 1.0,
                    "source_num_gpus": 1,
                },
                {
                    "source_duration_s": 160.0,
                    "source_interarrival_s": 1.0,
                    "source_num_gpus": 1,
                },
            ]
        )

        candidate = build_representativeness_profile(
            candidate_jobs,
            supported_gpu_counts=(1, 2),
        )

        score = score_representativeness(
            candidate,
            target,
        )

        self.assertGreater(score["total_score"], 0.0)
        self.assertGreater(
            score["components"][
                "gpu_demand_distance"
            ],
            0.0,
        )
        self.assertGreater(
            score["components"][
                "runtime_cdf_distance"
            ],
            0.0,
        )

class RepresentativeWindowSelectionTests(
    unittest.TestCase
):
    def test_selects_best_matching_window(self) -> None:
        rows = []

        durations = [
            100,
            200,
            700,
            800,
            4000,
            5000,
            100,
            200,
            700,
            800,
            4000,
            5000,
        ]

        gpu_counts = [
            1,
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            2,
            2,
            1,
            1,
        ]

        submit_times = [
            0,
            10,
            20,
            30,
            40,
            50,
            200,
            210,
            220,
            230,
            240,
            250,
        ]

        for index, (
            duration,
            gpu_count,
            submit_time,
        ) in enumerate(
            zip(
                durations,
                gpu_counts,
                submit_times,
            )
        ):
            rows.append(
                {
                    "source_job_id": f"job-{index}",
                    "source_duration_s": duration,
                    "source_num_gpus": gpu_count,
                    "submit_time_s": submit_time,
                    "source_interarrival_s": (
                        0.0
                        if index == 0
                        else submit_time
                        - submit_times[index - 1]
                    ),
                }
            )

        jobs = pd.DataFrame(rows)

        window, metadata = (
            select_representative_job_window(
                jobs,
                num_jobs=6,
                supported_gpu_counts=(1, 2),
                minimum_jobs_by_gpu_count={2: 1},
            )
        )

        self.assertEqual(len(window), 6)
        self.assertGreaterEqual(
            int(
                (
                    window["source_num_gpus"] == 2
                ).sum()
            ),
            1,
        )

        self.assertEqual(
            metadata["selection_method"],
            (
                "best_representative_contiguous_"
                "eligible_window"
            ),
        )

        self.assertGreater(
            metadata["candidate_window_count"],
            0,
        )

        self.assertGreaterEqual(
            metadata["representativeness"][
                "total_score"
            ],
            0.0,
        )

        self.assertEqual(
            window.iloc[0]["submit_time_s"],
            0.0,
        )

    def test_rejects_impossible_minimum(self) -> None:
        jobs = pd.DataFrame(
            [
                {
                    "source_job_id": "job-1",
                    "source_duration_s": 100.0,
                    "source_num_gpus": 1,
                    "submit_time_s": 0.0,
                    "source_interarrival_s": 0.0,
                },
                {
                    "source_job_id": "job-2",
                    "source_duration_s": 200.0,
                    "source_num_gpus": 1,
                    "submit_time_s": 10.0,
                    "source_interarrival_s": 10.0,
                },
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "No representative window satisfies",
        ):
            select_representative_job_window(
                jobs,
                num_jobs=2,
                supported_gpu_counts=(1, 2),
                minimum_jobs_by_gpu_count={2: 1},
            )

    def test_constraint_survives_filtered_index_gaps(self) -> None:
        jobs = pd.DataFrame(
            [
                {
                    "source_job_id": "job-0",
                    "source_duration_s": 100.0,
                    "source_num_gpus": 1,
                    "submit_time_s": 0.0,
                    "source_interarrival_s": 0.0,
                },
                {
                    # Unsupported job creates a gap in the original index.
                    "source_job_id": "job-1",
                    "source_duration_s": 200.0,
                    "source_num_gpus": 8,
                    "submit_time_s": 10.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    "source_job_id": "job-2",
                    "source_duration_s": 300.0,
                    "source_num_gpus": 1,
                    "submit_time_s": 20.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    "source_job_id": "job-3",
                    "source_duration_s": 400.0,
                    "source_num_gpus": 2,
                    "submit_time_s": 30.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    # Another unsupported job creates another index gap.
                    "source_job_id": "job-4",
                    "source_duration_s": 500.0,
                    "source_num_gpus": 4,
                    "submit_time_s": 40.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    "source_job_id": "job-5",
                    "source_duration_s": 600.0,
                    "source_num_gpus": 1,
                    "submit_time_s": 50.0,
                    "source_interarrival_s": 10.0,
                },
                {
                    "source_job_id": "job-6",
                    "source_duration_s": 700.0,
                    "source_num_gpus": 1,
                    "submit_time_s": 60.0,
                    "source_interarrival_s": 10.0,
                },
            ]
        )

        window, metadata = select_representative_job_window(
            jobs,
            num_jobs=3,
            supported_gpu_counts=(1, 2),
            minimum_jobs_by_gpu_count={2: 1},
        )

        two_gpu_count = int(
            (window["source_num_gpus"] == 2).sum()
        )

        self.assertGreaterEqual(two_gpu_count, 1)

        self.assertGreaterEqual(
            int(
                metadata["selected_jobs_by_gpu_count"].get(
                    "2",
                    0,
                )
            ),
            1,
        )

        self.assertEqual(
            metadata["minimum_jobs_by_gpu_count"],
            {"2": 1},
        ) 

if __name__ == "__main__":
    unittest.main()
