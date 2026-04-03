from __future__ import annotations

import argparse
import asyncio
import json
from itertools import islice
from pathlib import Path

from evomdt.config import AppConfig, configure_logging
from evomdt.evaluation import aggregate_benchmark_results
from evomdt.io import iter_dataset_tasks, write_json
from evomdt.models import AgentExecutionTrace, AgentOutput, CaseRunResult, CoordinatorOutput
from evomdt.pipeline import EvoMDTSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first N benchmark tasks and print per-case traces.")
    parser.add_argument(
        "--input",
        default="data/samples/benchmarks/medagentsbench.jsonl",
        help="Path to the benchmark JSONL file.",
    )
    parser.add_argument("--config", default="config.toml", help="Path to the TOML config.")
    parser.add_argument("--limit", type=int, default=2, help="Number of cases to run.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start offset in the dataset.")
    parser.add_argument("--output", help="Optional path to save the aggregate summary JSON.")
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print a truncated raw JSON response for each agent.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not write per-run artifacts to the artifacts directory.",
    )
    return parser


def _truncate(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _question_preview(question: str) -> str:
    return _truncate(question, 320)


def _render_agent_trace(trace: AgentExecutionTrace, show_raw: bool) -> list[str]:
    output = trace.parsed_output
    lines = [f"[{trace.role}] model={trace.model_name} tokens={trace.usage.get('total_tokens', 'n/a')}"]

    if isinstance(output, AgentOutput):
        lines.append(f"  summary: {_truncate(output.summary)}")
        lines.append(f"  confidence: {output.confidence:.2f}")
        if output.structured_findings:
            findings = "; ".join(_truncate(item, 100) for item in output.structured_findings[:3])
            lines.append(f"  findings: {findings}")
        if output.candidate_actions:
            actions = "; ".join(
                _truncate(f"{action.stance}:{action.action}", 100) for action in output.candidate_actions[:3]
            )
            lines.append(f"  actions: {actions}")
        if output.risks_or_alerts:
            alerts = "; ".join(
                _truncate(f"{alert.severity}:{alert.concern}", 100) for alert in output.risks_or_alerts[:2]
            )
            lines.append(f"  alerts: {alerts}")
    elif isinstance(output, CoordinatorOutput):
        lines.append(f"  final_answer: {output.final_answer}")
        lines.append(f"  summary: {_truncate(output.summary)}")
        if output.accepted_actions:
            lines.append(f"  accepted: {'; '.join(_truncate(item, 100) for item in output.accepted_actions[:4])}")
        if output.audit_trace:
            lines.append(f"  audit: {_truncate(output.audit_trace[0], 180)}")

    if show_raw:
        lines.append(f"  raw: {_truncate(trace.raw_content, 500)}")

    return lines


def _render_case_result(index: int, result: CaseRunResult, show_raw: bool) -> str:
    lines = [
        f"=== Case {index}: {result.case.case_id} ===",
        f"Question: {_question_preview(result.case.question)}",
        f"Reference: {result.case.reference_answer or 'n/a'}",
        f"Final Answer: {result.coordinator_decision.final_answer}",
        f"Duration: {result.duration_seconds:.2f}s",
        f"Artifacts: {result.artifacts_dir}",
        "",
        "Agent Traces:",
    ]
    for trace in result.agent_traces:
        lines.extend(_render_agent_trace(trace, show_raw))
    lines.extend(
        [
            "",
            "Coordinator Decision:",
            f"  rationale: {_truncate(result.coordinator_decision.decision_rationale, 300)}",
            f"  accepted_actions: {'; '.join(_truncate(item, 100) for item in result.coordinator_decision.accepted_actions[:5]) or 'none'}",
            f"  rejected_actions: {'; '.join(_truncate(item, 100) for item in result.coordinator_decision.rejected_actions[:5]) or 'none'}",
        ]
    )
    if result.coordinator_decision.audit_trace:
        lines.append(f"  audit_trace[0]: {_truncate(result.coordinator_decision.audit_trace[0], 300)}")
    if result.evaluation is not None:
        metric_parts = [
            f"{key}={value:.4f}"
            for key, value in result.evaluation.metrics.items()
            if isinstance(value, float)
        ]
        lines.extend(
            [
                "",
                "Evaluation:",
                f"  metrics: {', '.join(metric_parts) if metric_parts else 'none'}",
                f"  feedback_tags: {', '.join(result.evaluation.feedback_tags) if result.evaluation.feedback_tags else 'none'}",
            ]
        )
    return "\n".join(lines)


def _select_cases(dataset_path: str | Path, start: int, limit: int) -> list:
    return list(islice(iter_dataset_tasks(dataset_path), start, start + limit))


async def run_subset(args: argparse.Namespace) -> dict:
    config = AppConfig.load(args.config)
    configure_logging(config)
    system = EvoMDTSystem(config)

    cases = _select_cases(args.input, args.start, args.limit)
    if not cases:
        raise ValueError(f"No cases found in {args.input} for start={args.start}, limit={args.limit}.")

    evaluations = []
    run_ids: list[str] = []

    print(f"Running {len(cases)} case(s) from {args.input} starting at offset {args.start}.\n")
    for offset, case in enumerate(cases, start=args.start + 1):
        result = await system.run_case(case, persist=not args.no_persist)
        run_ids.append(result.run_id)
        if result.evaluation is not None:
            evaluations.append(result.evaluation)
        print(_render_case_result(offset, result, args.show_raw))
        print()

    summary = {
        "cases": len(cases),
        "aggregate_metrics": aggregate_benchmark_results(evaluations),
        "run_ids": run_ids,
    }
    if args.output:
        write_json(args.output, summary)
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = asyncio.run(run_subset(args))
    print("=== Aggregate Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
