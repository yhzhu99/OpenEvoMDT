# OpenEvoMDT

An independent, framework-comparable reproduction of **EvoMDT**, the multi-agent oncology decision-making framework proposed in the paper *EvoMDT: a self-evolving multi-agent system for structured clinical decision-making in multi-cancer*.

This repository is designed for **fair comparison across agent frameworks**. It keeps the core multi-agent structure of EvoMDT while intentionally disabling retrieval by default, so benchmark results are not inflated by a private RAG stack or hidden document pipeline. Evidence and provenance are still exposed through explicit interfaces, which makes it possible to re-enable grounded variants later without changing the public contract.

## Paper Reference

**Original paper**

- Liu Q, Hu Z, Huang T, et al. *EvoMDT: a self-evolving multi-agent system for structured clinical decision-making in multi-cancer*. **npj Digital Medicine** (2026) 9:124.
- DOI: `10.1038/s41746-025-02304-8`
- Publisher page: <https://doi.org/10.1038/s41746-025-02304-8>
- PubMed Central: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12873204/>

**Paper-listed code repository**

- Repository link listed in the paper: <https://github.com/KesselZ/EvoMDT>
- Note: repository availability and contents can change over time. As checked on **2026-04-03**, the public repository exists, but its visible contents are relatively minimal and should not be treated as a complete reference implementation for reproduction or benchmarking.

## What This Repository Is

`OpenEvoMDT` is **not** the original authors' repository, and it should not be interpreted as a mirror of the paper-listed GitHub repo. It is a clean-room, Python-first reproduction focused on:

- preserving the **five-agent EvoMDT structure**
- making the system easier to inspect, test, and modify
- supporting **apples-to-apples comparison** with other agent frameworks
- keeping the execution contract explicit and reproducible

## Design Goals

This reproduction follows a specific philosophy:

- **Keep the paper core**: Diagnostic, Treatment, Safety, Monitoring, and Coordinator agents are all preserved.
- **Be fair in comparisons**: no RAG by default; no hidden retrieval corpus; no implicit adaptive state during evaluation runs.
- **Be auditable**: intermediate agent outputs, coordinator decisions, conflict records, and artifacts are saved explicitly.
- **Be implementation-friendly**: simple TOML config, small codebase, deterministic guardrails where they matter.

## Included vs Excluded

### Included

- five-agent workflow with an explicit `Coordinator` agent
- no-RAG prompts and structured message passing
- safety-first guardrail that blocks coordinator acceptance of explicitly vetoed actions
- benchmark-oriented evaluation:
  - `accuracy` for MCQ tasks
  - `BERTScore` for generation tasks
  - `weighted_plan_concordance`
  - `guideline_concordance` when the dataset provides structured guideline metadata
  - `safety_violation_rate`
  - `evidence_traceability_coverage`
- explicit evolution state and bounded prompt/weight adaptation

### Excluded by Default

- document retrieval
- guideline retrieval
- literature retrieval
- hidden private knowledge bases

Those parts were intentionally left out of the default runtime because they make cross-framework comparison less fair. The public schema still keeps provenance fields so retrieval-backed variants can be added later in a controlled way.

## Architecture

The current pipeline is:

1. `Diagnostic`, `Treatment`, `Safety`, and `Monitoring` agents run on the same structured case dossier.
2. A real `Coordinator` agent synthesizes those specialist outputs into one MDT-style plan.
3. A deterministic **safety guardrail** post-checks the coordinator result and removes actions that were explicitly vetoed by the Safety agent.
4. The run is evaluated and persisted as structured artifacts.

This gives you a setup that is still agentic, but much easier to compare fairly than a full hidden-RAG clinical stack.

## Why This Is Useful for Agent Framework Comparison

Most “medical agent” reproductions become hard to compare because the real performance signal is mixed with:

- custom retrieval corpora
- private prompt layers
- hidden weighting logic
- evaluation scripts that depend on opaque judges

`OpenEvoMDT` tries to reduce that noise.

If you want to compare frameworks such as LangGraph, AutoGen, CrewAI, OpenAI Agents, or custom orchestrators, this repository gives you:

- the same task shape
- the same role decomposition
- the same structured outputs
- the same artifact format
- the same fairness constraints

## Quick Start

### 1. Install dependencies

```bash
uv sync --python 3.12
```

### 2. Set an API key

At least one configured model backend must have a valid key:

```bash
export DEEPSEEK_API_KEY=...
```

### 3. Run a sample case

```bash
uv run python main.py run-case \
  --input data/samples/cases/sample_hcc_case.json
```

### 4. Run the sample benchmark

```bash
uv run python main.py run-benchmark \
  --input data/samples/benchmarks/sample_benchmark.jsonl
```

### 5. Run bounded evolution on a dataset

```bash
uv run python main.py evolve \
  --input data/samples/benchmarks/sample_benchmark.jsonl
```

### 6. Inspect a saved run

```bash
uv run python main.py inspect-run --run-id <run-id>
```

## Configuration

Configuration lives in [`config.toml`](./config.toml).

Key defaults:

- all five agents are configured explicitly
- the four specialist agents have base weights
- the coordinator has no vote weight of its own because it synthesizes rather than votes
- evaluation runs are **frozen by default**
- `BERTScore` is opt-in
- evolution state is stored in `artifacts/evolution_state.json`

## Repository Layout

- [`main.py`](./main.py): simple entrypoint for running cases, benchmarks, evolution, and run inspection
- [`config.toml`](./config.toml): LLM, runtime, agent, evaluation, and evolution configuration
- [`evomdt/`](./evomdt): core framework code
- [`data/samples/`](./data/samples): sample oncology cases and benchmark inputs
- [`tests/`](./tests): fake-provider tests that run without external APIs
- [`paper/EvoMDT.md`](./paper/EvoMDT.md): local paper copy used during reproduction and alignment

## Reproducibility Notes

For fairness, this repository makes two opinionated choices:

- **No implicit retrieval**
  Retrieval is not part of the default execution path.

- **No implicit adaptive evaluation**
  Standard case and benchmark runs do not silently load prior evolved state.

This is deliberate. The goal is to make comparisons reflect the agent workflow itself, not hidden supporting infrastructure.

## Current Status

The project currently supports:

- structured case execution
- benchmark execution from JSONL
- persisted artifacts per run
- bounded prompt/weight evolution
- unit tests with a fake provider

The current implementation should be treated as a **research reproduction / benchmarking scaffold**, not a clinical system.

## Citation

If you use this repository in research, please cite the original EvoMDT paper first.

```bibtex
@article{liu2026evomdt,
  title   = {EvoMDT: a self-evolving multi-agent system for structured clinical decision-making in multi-cancer},
  author  = {Liu, Qicai and Hu, Zhichao and Huang, Tao and others},
  journal = {npj Digital Medicine},
  volume  = {9},
  pages   = {124},
  year    = {2026},
  doi     = {10.1038/s41746-025-02304-8}
}
```

If you want to cite this reproduction as well, add a repository citation after the project metadata is finalized.

## Disclaimer

This repository is for **research, reproduction, and framework evaluation** only.

- It is **not** a medical device.
- It does **not** provide clinical advice.
- It should not be used for real patient care or treatment decisions.

## Acknowledgement

This project stands on the original EvoMDT paper and the public repository link cited by that paper. The goal here is not to replace the authors' work, but to provide a transparent reproduction that is easier to benchmark, inspect, and compare across agent frameworks.
