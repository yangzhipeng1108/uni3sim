"""Price every observed scheduler step twice: exact capture and real buckets.

Graph bucket rounding is a property of the runtime's capture set, not of the
schedule.  This diagnostic therefore holds the schedule completely fixed --
the same observed requests, the same logical work, the same step order -- and
varies only the capture set given to ``GraphExecutionConfig``.  Two
``ServingExecutionContext`` instances evaluate each step, one whose capture
sizes contain every integer (so bucket padding is identically zero) and one
carrying the hardware's real capture buckets.

The signed per-step difference ``T_padded - T_exact`` is then attributable to
padding alone.  A negative difference means the model priced more physical
rows as less work, which no shape can do by itself, so each one is reported
with the per-kernel decomposition needed to name its mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.serving.execution import ServingExecutionContext
from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
from src.serving.workload.graph_padding import GraphExecutionConfig
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import RuntimeInputs

from evaluation.serving.metrics import compact_step_alignment
from evaluation.serving.metrics import sharegpt_compact_evaluation as compact


SCHEMA_VERSION = 1


def _mapping(value: object, owner: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    return value


def _sequence(value: object, owner: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{owner} must be a sequence")
    return value


def _non_negative_int(value: object, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner} must be an integer")
    if value < 0:
        raise ValueError(f"{owner} must be non-negative")
    return value


def captured_schedule_steps(case: object) -> Iterable[ScheduleStep]:
    """Rebuild the observed scheduler steps from a compact case."""

    if case.scheduler_steps is None:
        raise ValueError(
            "paired-step diagnosis requires schema-v3 scheduler_steps")
    computed = [request.cached_hit_tokens for request in case.requests]
    for raw_step in case.scheduler_steps:
        step = _mapping(raw_step, "captured scheduler step")
        step_index = _non_negative_int(step["step_index"], "step_index")
        requests = []
        for raw_work in _sequence(
                step["scheduled_requests"], "captured scheduled requests"):
            work = _mapping(raw_work, "captured scheduled request")
            request_index = _non_negative_int(
                work["request_idx"], "captured request_idx")
            if request_index >= len(case.requests):
                raise ValueError("captured request_idx exceeds requests.csv")
            prefill_tokens = _non_negative_int(
                work["prefill_tokens"], "captured prefill_tokens")
            decode_tokens = _non_negative_int(
                work["decode_tokens"], "captured decode_tokens")
            phase = "prefill" if prefill_tokens else "decode"
            request = case.requests[request_index]
            requests.append(ScheduledRequest(
                request_id=f"request-{request_index:012d}",
                phase=phase,
                input_tokens=prefill_tokens,
                output_tokens=decode_tokens,
                computed_tokens=computed[request_index],
                trace_prefix_hit_tokens=request.cached_hit_tokens,
                scheduler_prefix_hit_tokens=request.cached_hit_tokens,
            ))
            computed[request_index] += prefill_tokens + decode_tokens
        if not requests:
            continue
        yield ScheduleStep(
            f"observed:{step_index}",
            "main",
            tuple(requests),
            attention_kv_layout=None,
        )


@dataclass(frozen=True)
class PairedCase:
    """Everything one case needs to be priced under two capture sets."""

    cell_id: str
    hardware_id: str
    model: object
    deployment: object
    hardware: object
    launch_policy: str
    prediction_method: str
    routing: object
    attention_kv_layout: str
    attention_kv_distribution: str
    capture_sizes: tuple[int, ...]
    configured_mode: str
    uniform_decode_query_len: int
    piecewise_eager_workload_kinds: tuple[str, ...]


def load_paired_case(
    repository_root: Path,
    config: object,
    cell: object,
) -> PairedCase:
    model_path = repository_root / cell.model.simulation_model_path
    hardware_path = repository_root / cell.hardware.hardware_config_path
    model = load_model_config(str(model_path))
    hardware, prediction, deployment = (
        compact_step_alignment.load_case_setup(
            config, model, hardware_path))

    graph = cell.hardware.graph_execution
    return PairedCase(
        cell_id=cell.cell_id,
        hardware_id=hardware.hardware_id,
        model=model,
        deployment=deployment,
        hardware=hardware,
        launch_policy=prediction["launch_policy"],
        prediction_method=prediction["prediction_method"],
        routing=compact._expected_expert_routing(repository_root, cell),
        attention_kv_layout=prediction["attention_kv_layout"],
        attention_kv_distribution=prediction["attention_kv_distribution"],
        capture_sizes=tuple(graph.capture_sizes),
        configured_mode=graph.configured_mode,
        uniform_decode_query_len=graph.uniform_decode_query_len,
        piecewise_eager_workload_kinds=tuple(
            graph.piecewise_eager_workload_kinds),
    )


def _graph_config(case: PairedCase, capture_sizes) -> GraphExecutionConfig:
    return GraphExecutionConfig(
        case.configured_mode,
        tuple(capture_sizes),
        case.piecewise_eager_workload_kinds,
        case.uniform_decode_query_len,
    )


def _context(case: PairedCase, capture_sizes) -> ServingExecutionContext:
    runtime_inputs = RuntimeInputs(
        expert_routing=case.routing,
        attention_kv_layout=case.attention_kv_layout,
        attention_kv_distribution=case.attention_kv_distribution,
        graph_execution=_graph_config(case, capture_sizes),
    )
    return ServingExecutionContext(
        case.model,
        case.deployment,
        {case.hardware_id: case.hardware},
        case.launch_policy,
        case.prediction_method,
        runtime_inputs,
        execution_mode="accelerated",
    )


def _launch_facts(spec) -> dict[str, object] | None:
    launch = spec.launch
    if launch is None:
        # Transfer calls have no compute-kernel launch configuration.
        return None
    return {
        "grid": list(launch.grid),
        "block": list(launch.block),
        "execution_variant": launch.execution_variant,
        "source_program": (
            None if launch.source_program is None
            else type(launch.source_program).__name__),
    }


def _kernel_key(call) -> str:
    spec = call.kernel_spec
    return f"{spec.kind.value}/{spec.variant}"


def _call_rows(components) -> tuple[dict[str, dict[str, object]], float]:
    """Aggregate one step's calls by kernel identity."""

    rows: dict[str, dict[str, object]] = {}
    total_s = 0.0
    occurrences = dict(components.execution_timeline.call_occurrence_counts)
    for call in components.kernel_plan.calls:
        prediction = components.predictions[call.call_id]
        repeats = occurrences[call.call_id] if call.call_id in occurrences else 1
        duration_s = prediction.t_total_s * repeats
        total_s += duration_s
        key = _kernel_key(call)
        if key not in rows:
            spec = call.kernel_spec
            attributes = spec.attributes
            rows[key] = {
                "kernel": key,
                "calls": 0,
                "device_time_ms": 0.0,
                "shape": dict(spec.shape),
                "graph_runtime_mode": (
                    attributes["graph_runtime_mode"]
                    if "graph_runtime_mode" in attributes else None),
                "graph_logical_tokens": (
                    attributes["graph_logical_tokens"]
                    if "graph_logical_tokens" in attributes else None),
                "graph_model_input_tokens": (
                    attributes["graph_model_input_tokens"]
                    if "graph_model_input_tokens" in attributes else None),
                "launch": _launch_facts(spec),
                "bottleneck_group": prediction.bottleneck_group,
                "limitations": sorted(
                    () if prediction.diagnostics is None
                    else prediction.diagnostics.limitations),
            }
        rows[key]["calls"] += repeats
        rows[key]["device_time_ms"] += duration_s * 1e3
    return rows, total_s


def _critical_path(components) -> list[list[object]]:
    contributions = components.execution_timeline.critical_path_contributions
    ranked = sorted(contributions, key=lambda item: -item[1])[:8]
    return [[name, value * 1e3] for name, value in ranked]


def _step_row(
    step_index: int,
    schedule: ScheduleStep,
    exact_components,
    padded_components,
    exact_shape,
    padded_shape,
) -> dict[str, object]:
    exact_rows, exact_total_s = _call_rows(exact_components)
    padded_rows, padded_total_s = _call_rows(padded_components)
    kernels = []
    for key in sorted(set(exact_rows) | set(padded_rows)):
        exact_row = exact_rows[key] if key in exact_rows else None
        padded_row = padded_rows[key] if key in padded_rows else None
        exact_ms = 0.0 if exact_row is None else exact_row["device_time_ms"]
        padded_ms = 0.0 if padded_row is None else padded_row["device_time_ms"]
        kernels.append({
            "kernel": key,
            "exact": exact_row,
            "padded": padded_row,
            "delta_ms": padded_ms - exact_ms,
        })
    kernels.sort(key=lambda row: row["delta_ms"])
    phases = sorted({request.phase for request in schedule.requests})
    return {
        "step_index": step_index,
        "step_id": schedule.step_id,
        "phases": phases,
        "batch": len(schedule.requests),
        "logical_tokens": schedule.token_count,
        "exact_physical_tokens": exact_shape.model_input_tokens,
        "padded_physical_tokens": padded_shape.model_input_tokens,
        "padding_tokens": padded_shape.padding_tokens,
        "exact_graph_mode": exact_shape.runtime_mode,
        "padded_graph_mode": padded_shape.runtime_mode,
        "exact_total_ms": exact_total_s * 1e3,
        "padded_total_ms": padded_total_s * 1e3,
        "delta_ms": (padded_total_s - exact_total_s) * 1e3,
        "exact_timeline_ms": exact_components.execution_timeline.duration_s * 1e3,
        "padded_timeline_ms": (
            padded_components.execution_timeline.duration_s * 1e3),
        "exact_critical_path_ms": _critical_path(exact_components),
        "padded_critical_path_ms": _critical_path(padded_components),
        "kernels": kernels,
    }


def run_paired_steps(
    repository_root: Path,
    config: object,
    cell: object,
    case: object,
    *,
    step_limit: int | None = None,
) -> dict[str, object]:
    """Price every observed step under exact and real capture sets."""

    paired = load_paired_case(repository_root, config, cell)
    exact_capture = tuple(range(1, max(paired.capture_sizes) + 1))
    exact_context = _context(paired, exact_capture)
    padded_context = _context(paired, paired.capture_sizes)
    exact_graph = exact_context.runtime_inputs.graph_execution
    padded_graph = padded_context.runtime_inputs.graph_execution

    steps = []
    for step_index, schedule in enumerate(captured_schedule_steps(case)):
        if step_limit is not None and step_index >= step_limit:
            break
        uniform_decode = all(
            request.scheduled_tokens == paired.uniform_decode_query_len
            for request in schedule.requests)
        exact_shape = exact_graph.resolve(
            schedule.token_count, uniform_decode=uniform_decode)
        padded_shape = padded_graph.resolve(
            schedule.token_count, uniform_decode=uniform_decode)
        exact_components = exact_context.execute_step(schedule)
        padded_components = padded_context.execute_step(schedule)
        steps.append(_step_row(
            step_index,
            schedule,
            exact_components,
            padded_components,
            exact_shape,
            padded_shape,
        ))
        del exact_components, padded_components

    padded_steps = [row for row in steps if row["padding_tokens"] > 0]
    negative_steps = [row for row in steps if row["delta_ms"] < 0.0]
    negative_padded = [row for row in padded_steps if row["delta_ms"] < 0.0]
    kernel_totals: dict[str, float] = {}
    negative_kernel_totals: dict[str, float] = {}
    for row in steps:
        for kernel in row["kernels"]:
            name = kernel["kernel"]
            kernel_totals[name] = (
                (kernel_totals[name] if name in kernel_totals else 0.0)
                + kernel["delta_ms"])
            if kernel["delta_ms"] < 0.0:
                negative_kernel_totals[name] = (
                    (negative_kernel_totals[name]
                     if name in negative_kernel_totals else 0.0)
                    + kernel["delta_ms"])
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_id": paired.cell_id,
        "hardware_id": paired.hardware_id,
        "configured_graph_mode": paired.configured_mode,
        "capture_sizes": list(paired.capture_sizes),
        "exact_capture_sizes_max": exact_capture[-1],
        "summary": {
            "steps": len(steps),
            "padded_steps": len(padded_steps),
            "negative_steps": len(negative_steps),
            "negative_padded_steps": len(negative_padded),
            "sum_delta_ms": sum(row["delta_ms"] for row in steps),
            "sum_padded_delta_ms": sum(
                row["delta_ms"] for row in padded_steps),
            "sum_negative_delta_ms": sum(
                row["delta_ms"] for row in negative_steps),
            "kernel_delta_ms": dict(sorted(
                kernel_totals.items(), key=lambda item: item[1])),
            "negative_kernel_delta_ms": dict(sorted(
                negative_kernel_totals.items(), key=lambda item: item[1])),
        },
        "steps": steps,
    }


def write_report(report: Mapping[str, object], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "graph_padding_paired_steps.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path
