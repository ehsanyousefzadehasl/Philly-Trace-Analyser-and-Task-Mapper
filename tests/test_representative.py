import unittest

import pandas as pd

from trace_mapper.representative import (
    build_representativeness_profile,
    score_representativeness,
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


if __name__ == "__main__":
    unittest.main()
