# OpenEvoMDT

Python reproduction of the EvoMDT paper with:

- a fixed five-agent oncology workflow with an explicit coordinator agent
- TOML-based LLM configuration
- frozen-by-default evaluation runs and explicit evolution state
- sample benchmark and case-evaluation flows

For fair framework comparison, the default runtime is no-RAG and does not implicitly load adaptive state from prior runs. The coordinator is a fifth LLM agent, while a deterministic safety guardrail still blocks actions explicitly vetoed by the Safety agent.

The project uses `uv` with Python 3.12 and a simple `main.py` entrypoint instead of an installed CLI.

## Setup

```bash
uv sync --python 3.12
```

Set one of the configured API keys before running real model calls:

```bash
export DEEPSEEK_API_KEY=...
```

## Usage

Run a single case:

```bash
uv run python main.py run-case --input data/samples/cases/sample_hcc_case.json
```

Run the sample benchmark:

```bash
uv run python main.py run-benchmark --input data/samples/benchmarks/sample_benchmark.jsonl
```

`run-benchmark` reads the input as JSONL, with one task/case per line.

Apply bounded prompt/weight evolution on a case or dataset:

```bash
uv run python main.py evolve --input data/samples/benchmarks/sample_benchmark.jsonl
```

Inspect a saved run:

```bash
uv run python main.py inspect-run --run-id <run-id>
```

## Layout

- `main.py`: simple entrypoint for running cases, benchmarks, evolution, and run inspection
- `config.toml`: HealthFlow-style LLM and runtime configuration
- `evomdt/`: framework code for config, LLM calls, prompts, five-agent coordination, evaluation, and evolution
- `data/samples/`: example oncology inputs for local smoke tests
- `tests/`: fake-provider unit tests that do not call external APIs
