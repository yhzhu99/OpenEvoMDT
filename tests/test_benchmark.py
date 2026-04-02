import asyncio
from pathlib import Path

from evomdt.pipeline import EvoMDTSystem

from .conftest import FakeProvider


def test_run_benchmark_writes_summary(test_config):
    system = EvoMDTSystem(test_config, provider_factory=lambda model_key: FakeProvider())
    output_path = Path(test_config.runtime.artifacts_dir) / "summary.json"
    summary = asyncio.run(system.run_benchmark("data/samples/benchmarks/sample_benchmark.jsonl", output_path=output_path))
    assert summary["cases"] == 2
    assert output_path.exists()
    assert "composite_dimension_mean" in summary["aggregate_metrics"]
