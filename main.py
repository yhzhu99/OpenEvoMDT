from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenEvoMDT runner")
    parser.add_argument(
        "command",
        choices=["run-case", "run-benchmark", "evolve", "inspect-run"],
        help="Operation to execute.",
    )
    parser.add_argument("--config", default="config.toml", help="Path to the TOML config.")
    parser.add_argument("--input", help="Path to a case JSON or benchmark JSONL file.")
    parser.add_argument("--run-id", help="Run identifier for inspect-run.")
    parser.add_argument("--output", help="Optional output path for results.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    print(
        "Project scaffold created. Runtime implementation is provided in the evomdt package. "
        "Use `python main.py --help` to inspect available commands."
    )


if __name__ == "__main__":
    main()
