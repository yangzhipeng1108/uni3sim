"""Evaluation-only first-schedule membership capture for a paired virtual case."""

import json
from pathlib import Path

from src.serving.backend.vllm import VLLMBackend

from .symbolic_model import run_paired_virtual_case


def run_first_schedule_probe(simulation_argv, case, comparison_output, trace_output):
    """Record first-admission step members without retaining kernel call payloads."""
    comparison_output = Path(comparison_output)
    trace_output = Path(trace_output)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    original = VLLMBackend.plan_step
    first_request_ids = set()
    first_steps = []
    step_count = 0

    def record(self, virtual_time_s):
        nonlocal step_count
        step = original(self, virtual_time_s)
        if step is None:
            return None
        requests = step.schedule.requests
        indexes = []
        for request in requests:
            prefix = "request-"
            if (not request.request_id.startswith(prefix)
                    or len(request.request_id) != len(prefix) + 12
                    or not request.request_id[len(prefix):].isdigit()):
                raise ValueError("first-schedule probe requires canonical request IDs")
            indexes.append(int(request.request_id[len(prefix):]))
        newcomers = [
            index for request, index in zip(requests, indexes, strict=True)
            if request.request_id not in first_request_ids
        ]
        if newcomers:
            first_steps.append({
                "step_index": step_count,
                "step_id": step.schedule.step_id,
                "virtual_time_s": virtual_time_s,
                "newcomer_indexes": newcomers,
                "request_indexes": indexes,
                "request_phases": [request.phase for request in requests],
                "request_token_counts": [request.scheduled_tokens for request in requests],
            })
        first_request_ids.update(request.request_id for request in requests)
        step_count += 1
        return step

    VLLMBackend.plan_step = record
    try:
        run_paired_virtual_case(simulation_argv, case, comparison_output)
    finally:
        VLLMBackend.plan_step = original
    if len(first_request_ids) != len(case.requests):
        raise ValueError("first-schedule probe did not cover every canonical request")
    trace_output.write_text(json.dumps({
        "schema_version": 1,
        "analysis_kind": "evaluation_only_first_schedule_membership",
        "request_source_sha256": case.manifest["requests"]["sha256"],
        "step_count": step_count,
        "first_request_count": len(first_request_ids),
        "first_steps": first_steps,
    }, indent=2) + "\n")
