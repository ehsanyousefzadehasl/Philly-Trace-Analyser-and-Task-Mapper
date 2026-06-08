from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trace_mapper.config import load_generation_config


import json

class GenerationConfigTests(unittest.TestCase):
    def test_load_valid_config(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"

            config_path.write_text(
                """
version: 1

source:
  format: philly
  trace_path: source.csv
  cluster_name: philly

mapping:
  workload_catalog_path: catalog.csv
  seed: 7
  num_jobs: 60
  supported_gpu_counts: [1, 2]
  gpu_demand_matching: exact
  duration_matching: nearest_quantile
  unsupported_gpu_policy: filter
  preserve_arrivals: true
  minimum_jobs_by_gpu_count:
    2: 2
  selection_method: representative

output:
  trace_path: mapped.csv
  manifest_path: mapped.manifest.json
""".strip(),
                encoding="utf-8",
            )

            config = load_generation_config(config_path)

            self.assertEqual(config.source.format, "philly")
            self.assertEqual(
                config.source.trace_path,
                Path("source.csv"),
            )
            self.assertEqual(config.source.cluster_name, "philly")

            self.assertEqual(config.mapping.seed, 7)
            self.assertEqual(config.mapping.num_jobs, 60)
            self.assertEqual(
                config.mapping.supported_gpu_counts,
                (1, 2),
            )
            self.assertEqual(
                config.mapping.minimum_jobs_by_gpu_count,
                ((2, 2),),
            )
            self.assertTrue(config.mapping.preserve_arrivals)

            self.assertEqual(
                config.mapping.selection_method,
                "representative",
            )

            self.assertEqual(
                config.output.trace_path,
                Path("mapped.csv"),
            )

    def test_rejects_unknown_selection_method(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"

            config = {
                "version": 1,
                "source": {
                    "format": "philly",
                    "trace_path": "trace.json",
                    "cluster_name": "philly",
                },
                "mapping": {
                    "workload_catalog_path": "catalog.csv",
                    "seed": 42,
                    "num_jobs": 10,
                    "selection_method": "unknown_method",
                    "supported_gpu_counts": [1, 2],
                    "gpu_demand_matching": "exact",
                    "duration_matching": "nearest_quantile",
                    "unsupported_gpu_policy": "filter",
                    "preserve_arrivals": True,
                },
                "output": {
                    "trace_path": "mapped.csv",
                    "manifest_path": "mapped.manifest.json",
                },
            }

            config_path.write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "mapping.selection_method",
            ):
                load_generation_config(config_path)
    def test_reject_empty_gpu_counts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"

            config_path.write_text(
                """
version: 1
source:
  format: philly
  trace_path: source.csv
mapping:
  workload_catalog_path: catalog.csv
  supported_gpu_counts: []
output:
  trace_path: mapped.csv
  manifest_path: mapped.json
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "supported_gpu_counts",
            ):
                load_generation_config(config_path)
    
if __name__ == "__main__":
    unittest.main()