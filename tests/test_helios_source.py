from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from trace_mapper.sources.helios import load_helios_jobs


class HeliosSourceTests(unittest.TestCase):
    def test_normalizes_completed_gpu_jobs(self) -> None:
        rows = [
            {
                "job_id": 2,
                "gpu_num": 2,
                "node_num": 1,
                "state": "COMPLETED",
                "submit_time": "2020-01-01 00:00:10",
                "start_time": "2020-01-01 00:00:20",
                "end_time": "2020-01-01 00:01:20",
                "duration": 60,
                "queue": 10,
            },
            {
                "job_id": 1,
                "gpu_num": 1,
                "node_num": 1,
                "state": "COMPLETED",
                "submit_time": "2020-01-01 00:00:00",
                "start_time": "2020-01-01 00:00:05",
                "end_time": "2020-01-01 00:00:35",
                "duration": 30,
                "queue": 5,
            },
            {
                "job_id": 3,
                "gpu_num": 0,
                "node_num": 1,
                "state": "COMPLETED",
                "submit_time": "2020-01-01 00:00:00",
                "start_time": "2020-01-01 00:00:00",
                "end_time": "2020-01-01 00:00:10",
                "duration": 10,
                "queue": 0,
            },
            {
                "job_id": 4,
                "gpu_num": 1,
                "node_num": 1,
                "state": "FAILED",
                "submit_time": "2020-01-01 00:00:00",
                "start_time": "2020-01-01 00:00:00",
                "end_time": "2020-01-01 00:00:10",
                "duration": 10,
                "queue": 0,
            },
        ]

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cluster_log.csv"
            pd.DataFrame(rows).to_csv(path, index=False)

            result = load_helios_jobs(
                path,
                cluster_name="saturn",
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            result["source_job_id"].tolist(),
            ["1", "2"],
        )

        first = result.iloc[0]
        second = result.iloc[1]

        self.assertEqual(first["source_trace"], "helios")
        self.assertEqual(first["source_cluster"], "saturn")
        self.assertEqual(first["source_num_gpus"], 1)
        self.assertEqual(first["source_duration_s"], 30.0)
        self.assertEqual(first["source_queue_wait_s"], 5.0)
        self.assertEqual(first["submit_time_s"], 0.0)
        self.assertEqual(first["source_interarrival_s"], 0.0)

        self.assertEqual(second["source_num_gpus"], 2)
        self.assertEqual(second["source_duration_s"], 60.0)
        self.assertEqual(second["source_queue_wait_s"], 10.0)
        self.assertEqual(second["submit_time_s"], 10.0)
        self.assertEqual(second["source_interarrival_s"], 10.0)

    def test_rejects_timestamp_duration_mismatch(self) -> None:
        rows = [
            {
                "job_id": 1,
                "gpu_num": 1,
                "node_num": 1,
                "state": "COMPLETED",
                "submit_time": "2020-01-01 00:00:00",
                "start_time": "2020-01-01 00:00:05",
                "end_time": "2020-01-01 00:00:35",
                "duration": 999,
                "queue": 5,
            }
        ]

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cluster_log.csv"
            pd.DataFrame(rows).to_csv(path, index=False)

            with self.assertRaisesRegex(
                ValueError,
                "duration mismatches",
            ):
                load_helios_jobs(
                    path,
                    cluster_name="venus",
                )


if __name__ == "__main__":
    unittest.main()