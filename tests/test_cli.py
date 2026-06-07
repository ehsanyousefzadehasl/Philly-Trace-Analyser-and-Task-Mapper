import unittest

from trace_mapper.cli import build_parser


class CliTests(unittest.TestCase):
    def test_build_catalog_command_parses(self) -> None:
        args = build_parser().parse_args(
            [
                "build-catalog",
                "--profiles",
                "profiles_1gpu.csv",
                "profiles_2gpu.csv",
                "--output",
                "workload_catalog.csv",
                "--task-root",
                "/workloads",
            ]
        )

        self.assertEqual(args.command, "build-catalog")
        self.assertEqual(
            [str(path) for path in args.profiles],
            ["profiles_1gpu.csv", "profiles_2gpu.csv"],
        )
        self.assertEqual(str(args.output), "workload_catalog.csv")
        self.assertEqual(str(args.task_root), "/workloads")

    def test_generate_command_parses(self) -> None:
        args = build_parser().parse_args(["generate"])
        self.assertEqual(args.command, "generate")


if __name__ == "__main__":
    unittest.main()