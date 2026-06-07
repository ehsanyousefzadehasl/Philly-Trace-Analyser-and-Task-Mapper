import unittest

from trace_mapper.cli import build_parser


class CliTests(unittest.TestCase):
    def test_build_catalog_command_parses(self) -> None:
        args = build_parser().parse_args(["build-catalog"])
        self.assertEqual(args.command, "build-catalog")

    def test_generate_command_parses(self) -> None:
        args = build_parser().parse_args(["generate"])
        self.assertEqual(args.command, "generate")


if __name__ == "__main__":
    unittest.main()