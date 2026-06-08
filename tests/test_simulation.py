import unittest

import pandas as pd

from trace_mapper.simulation import (
    simulate_exclusive_execution,
)


class ExclusiveSimulationTests(unittest.TestCase):
    def test_simulates_one_and_two_gpu_jobs(self) -> None:
        trace = pd.DataFrame(
            [
                {
                    "submit_time_s": 0.0,
                    "task_path": "a.yaml",
                    "mapped_num_gpus": 1,
                    "mapped_solo_runtime_s": 100.0,
                },
                {
                    "submit_time_s": 0.0,
                    "task_path": "b.yaml",
                    "mapped_num_gpus": 1,
                    "mapped_solo_runtime_s": 50.0,
                },
                {
                    "submit_time_s": 10.0,
                    "task_path": "c.yaml",
                    "mapped_num_gpus": 2,
                    "mapped_solo_runtime_s": 30.0,
                },
            ]
        )

        simulated, summary = simulate_exclusive_execution(
            trace,
            server_gpu_count=3,
        )

        self.assertEqual(len(simulated), 3)

        first = simulated.iloc[0]
        second = simulated.iloc[1]
        third = simulated.iloc[2]

        self.assertEqual(
            first["exclusive_start_time_s"],
            0.0,
        )
        self.assertEqual(
            second["exclusive_start_time_s"],
            0.0,
        )

        # The 2-GPU job waits until two GPUs are available.
        self.assertEqual(
            third["exclusive_start_time_s"],
            50.0,
        )
        self.assertEqual(
            third["exclusive_end_time_s"],
            80.0,
        )
        self.assertEqual(
            third["exclusive_waiting_time_s"],
            40.0,
        )

        self.assertEqual(summary["server_gpu_count"], 3)
        self.assertEqual(summary["job_count"], 3)
        self.assertEqual(summary["makespan_s"], 100.0)

    def test_rejects_job_larger_than_server(self) -> None:
        trace = pd.DataFrame(
            [
                {
                    "submit_time_s": 0.0,
                    "task_path": "large.yaml",
                    "mapped_num_gpus": 4,
                    "mapped_solo_runtime_s": 100.0,
                }
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "more GPUs than the server",
        ):
            simulate_exclusive_execution(
                trace,
                server_gpu_count=3,
            )


if __name__ == "__main__":
    unittest.main()
