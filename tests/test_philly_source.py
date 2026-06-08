import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trace_mapper.sources.philly import load_philly_jobs


class PhillySourceTests(unittest.TestCase):
    def test_normalizes_valid_passed_jobs(self) -> None:
        jobs = [
            {
                "status": "Pass",
                "jobid": "job-2",
                "submitted_time": "2017-01-01 00:00:10",
                "attempts": [
                    {
                        "start_time": "2017-01-01 00:00:20",
                        "end_time": "2017-01-01 00:01:20",
                        "detail": [
                            {
                                "ip": "node-1",
                                "gpus": ["gpu0", "gpu1"],
                            }
                        ],
                    }
                ],
            },
            {
                "status": "Pass",
                "jobid": "job-1",
                "submitted_time": "2017-01-01 00:00:00",
                "attempts": [
                    {
                        "start_time": "2017-01-01 00:00:05",
                        "end_time": "2017-01-01 00:00:35",
                        "detail": [
                            {
                                "ip": "node-2",
                                "gpus": ["gpu0"],
                            }
                        ],
                    }
                ],
            },
            {
                "status": "Failed",
                "jobid": "failed-job",
                "submitted_time": "2017-01-01 00:00:00",
                "attempts": [],
            },
            {
                "status": "Pass",
                "jobid": "invalid-time",
                "submitted_time": "2017-01-01 00:00:00",
                "attempts": [
                    {
                        "start_time": "None",
                        "end_time": "None",
                        "detail": [],
                    }
                ],
            },
        ]

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cluster_job_log"
            path.write_text(json.dumps(jobs), encoding="utf-8")

            result = load_philly_jobs(path)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            result["source_job_id"].tolist(),
            ["job-1", "job-2"],
        )

        first = result.iloc[0]
        second = result.iloc[1]

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


if __name__ == "__main__":
    unittest.main()