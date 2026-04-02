from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from .artifacts import inspect_run as read_run_summary
from .artifacts import save_case_run
from .config import AppConfig
from .consensus import deterministic_coordinator
from .evaluation import aggregate_benchmark_results, evaluate_case
from .evolution import apply_feedback, load_state, save_state
from .io import iter_dataset_tasks, load_case, write_json
from .llm import ChatMessage, OpenAIChatProvider, StructuredLLM, parse_json_content
from .models import AgentExecutionTrace, AgentOutput, CaseDossier, CaseRunResult, EvolutionState, RoleName, SPECIALIST_ROLES, utc_timestamp
from .prompts import build_system_prompt, build_user_prompt

LOGGER = logging.getLogger("evomdt.pipeline")


class EvoMDTSystem:
    def __init__(
        self,
        config: AppConfig,
        provider_factory: Callable[[str], StructuredLLM] | None = None,
    ):
        self.config = config
        self.provider_factory = provider_factory or self._default_provider_factory

    def _default_provider_factory(self, model_key: str) -> StructuredLLM:
        return OpenAIChatProvider(self.config.llm_for(model_key))

    async def _run_specialist(self, case: CaseDossier, role: RoleName, refinement_ids: list[str]) -> AgentExecutionTrace:
        role_config = self.config.agents.roles[role]
        provider = self.provider_factory(role_config.llm)
        messages = [
            ChatMessage(role="system", content=build_system_prompt(role, refinement_ids)),
            ChatMessage(role="user", content=build_user_prompt(case, role)),
        ]
        trace = await provider.generate_structured(
            messages,
            parser=lambda content: AgentOutput.model_validate(parse_json_content(content)),
        )
        parsed_output = AgentOutput.model_validate(trace.parsed)
        if parsed_output.role != role:
            parsed_output.role = role
        return AgentExecutionTrace(
            role=role,
            model_name=trace.model_name,
            raw_content=trace.raw_content,
            usage=trace.usage,
            parsed_output=parsed_output,
        )

    async def run_case(
        self,
        case: CaseDossier,
        *,
        persist: bool = True,
        state: EvolutionState | None = None,
    ) -> CaseRunResult:
        state = state or load_state(self.config)
        started_at = perf_counter()
        refinement_ids_by_role = state.prompt_refinements

        if self.config.agents.parallel:
            traces = await asyncio.gather(
                *[
                    self._run_specialist(case, role, refinement_ids_by_role.get(role, []))
                    for role in SPECIALIST_ROLES
                ]
            )
        else:
            traces = []
            for role in SPECIALIST_ROLES:
                traces.append(await self._run_specialist(case, role, refinement_ids_by_role.get(role, [])))

        outputs = [trace.parsed_output for trace in traces]
        coordinator_decision = deterministic_coordinator(case, outputs, state.role_weights)
        evaluation = evaluate_case(case, outputs, coordinator_decision, self.config)
        timestamp = utc_timestamp().replace(":", "").replace("-", "").replace("+00:00", "Z")
        run_id = f"{timestamp}_{case.case_id.replace(' ', '-').lower()}"
        duration_seconds = round(perf_counter() - started_at, 4)
        result = CaseRunResult(
            run_id=run_id,
            case=case,
            agent_traces=traces,
            coordinator_decision=coordinator_decision,
            evaluation=evaluation,
            duration_seconds=duration_seconds,
            artifacts_dir=str(self.config.artifacts_dir / "runs" / run_id),
        )
        if persist:
            save_case_run(self.config, result)
        return result

    async def run_case_file(self, path: str | Path, *, persist: bool = True) -> CaseRunResult:
        return await self.run_case(load_case(path), persist=persist)

    async def run_benchmark(self, path: str | Path, *, output_path: str | Path | None = None) -> dict:
        results: list[CaseRunResult] = []
        evaluations = []
        for case in iter_dataset_tasks(path):
            case_result = await self.run_case(case)
            results.append(case_result)
            if case_result.evaluation is not None:
                evaluations.append(case_result.evaluation)
        summary = {
            "cases": len(results),
            "aggregate_metrics": aggregate_benchmark_results(evaluations),
            "run_ids": [result.run_id for result in results],
        }
        if output_path is not None:
            write_json(output_path, summary)
        else:
            write_json(self.config.artifacts_dir / "benchmark_summary.json", summary)
        return summary

    async def evolve(self, path: str | Path) -> dict:
        state = load_state(self.config)
        evolved_cases = 0
        if Path(path).suffix == ".jsonl":
            cases = iter_dataset_tasks(path)
        else:
            cases = [load_case(path)]
        for case in cases:
            result = await self.run_case(case, state=state)
            if result.evaluation is not None:
                state = apply_feedback(self.config, state, result.evaluation)
                evolved_cases += 1
        save_state(self.config, state)
        return {
            "updated_cases": evolved_cases,
            "role_weights": state.role_weights,
            "prompt_refinements": state.prompt_refinements,
            "history_entries": len(state.history),
        }

    def inspect_run(self, run_id: str) -> dict:
        return read_run_summary(self.config, run_id)
