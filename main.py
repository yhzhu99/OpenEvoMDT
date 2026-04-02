from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evomdt.config import AppConfig, configure_logging
from evomdt.pipeline import EvoMDTSystem


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


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    config = AppConfig.load(config_path)
    configure_logging(config)
    system = EvoMDTSystem(config)

    if args.command == "run-case":
        if not args.input:
            parser.error("--input is required for run-case")
        result = await system.run_case_file(args.input)
        if args.output:
            Path(args.output).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(result.coordinator_decision.response_text)
        print(f"\nRun ID: {result.run_id}")
        return

    if args.command == "run-benchmark":
        if not args.input:
            parser.error("--input is required for run-benchmark")
        summary = await system.run_benchmark(args.input, output_path=args.output)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "evolve":
        if not args.input:
            parser.error("--input is required for evolve")
        summary = await system.evolve(args.input)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "inspect-run":
        if not args.run_id:
            parser.error("--run-id is required for inspect-run")
        summary = system.inspect_run(args.run_id)
        print(json.dumps(summary, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    asyncio.run(async_main())
