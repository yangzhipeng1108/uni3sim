"""Serving-driven Kernel coverage census.

The primary path replays captured schema-v3 Scheduler steps.  A separate
predicted-Scheduler control path consumes a full execution artifact; the two
sources are never presented as equivalent evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re

from evaluation.serving.metrics import compact_step_alignment
from evaluation.serving.metrics import sharegpt_compact_evaluation as compact
from evaluation.serving.simulation import simulation_runner
from evaluation.serving.simulation.execution_artifact import (
    execution_step_components_manifest,
)
from src.kernel.dataset_storage import load_kernel_dataset_document
from src.serving.execution import ServingExecutionContext
from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import RuntimeInputs


SCHEMA_VERSION = 1
REPORT_KIND = "serving_kernel_coverage"
INPUT_MODES = ("captured_scheduler_replay", "predicted_scheduler_control")
_ARTIFACT_SCHEMA_VERSIONS = (5, 7, 8)
_COMPUTE_KINDS = {
    "GEMM", "FLASH_ATTN", "EAGER_ATTN", "SPARSE_ATTN_INDEXER",
    "SPARSE_ATTN", "KV_CACHE", "FUSED_MOE", "UNFUSED_MOE", "VECTOR",
}
_FAMILY_FILENAMES = {
    "gemm": "gemm.json.zst",
    "flash_attn": "flash_attn.json.zst",
    "fused_moe": "fused_moe.json.zst",
    "vector": "vector.json.zst",
}
_LAYER_NODE = re.compile(r"(?:^|:)layer-[0-9]+:(?P<role>[^:]+)$")


def parse_coverage_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the C500 coverage workflow arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", action="append",
        help="compact evaluation config; defaults to the three active suites",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--case", action="append",
        help="hardware/model/case_id; repeat for multiple cells",
    )
    selection.add_argument(
        "--all-cases", action="store_true",
        help="select every MetaX C500 cell across the supplied configs",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--candidate-root",
        help=(
            "optional compact root containing hardware/model/case_id; "
            "for recapture candidates use profiler_output/"
            "serving_dataset_compaction"
        ),
    )
    parser.add_argument(
        "--input-mode", choices=INPUT_MODES,
        default="captured_scheduler_replay",
    )
    parser.add_argument(
        "--execution-mode", choices=("accelerated", "reference"),
        default="accelerated",
    )
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 1, metavar="N",
    )
    return parser.parse_args(argv)


def select_coverage_cells(
    repository_root: Path,
    arguments: argparse.Namespace,
    default_configs: Sequence[str],
    hardware_id: str = "metax_c500",
) -> tuple[tuple[compact.Phase1Config, compact.Phase1Cell], ...]:
    """Resolve a complete, duplicate-free single-hardware CLI selection."""

    config_values = (
        tuple(default_configs)
        if arguments.config is None
        else tuple(arguments.config)
    )
    requested = () if arguments.case is None else tuple(arguments.case)
    selected = []
    selected_ids = set()
    for config_value in config_values:
        config_path = compact.resolve_repository_input(
            repository_root, config_value, "coverage config"
        )
        config = compact.load_phase1_config(config_path)
        all_cells = compact.select_phase1_cells(config, (), all_cases=True)
        if arguments.all_cases:
            cells = tuple(
                cell for cell in all_cells
                if cell.hardware.hardware_id == hardware_id
            )
        else:
            config_ids = {cell.cell_id for cell in all_cells}
            matching = tuple(
                cell_id for cell_id in requested if cell_id in config_ids
            )
            cells = (
                compact.select_phase1_cells(config, matching, all_cases=False)
                if matching
                else ()
            )
        for cell in cells:
            if cell.cell_id in selected_ids:
                raise ValueError(
                    f"configs select duplicate coverage cell {cell.cell_id!r}"
                )
            selected_ids.add(cell.cell_id)
            selected.append((config, cell))
    if not selected:
        raise ValueError(
            f"coverage selection contains no {hardware_id} cells")
    if not arguments.all_cases:
        missing = sorted(set(requested) - selected_ids)
        if missing:
            raise ValueError(f"unknown coverage cell IDs: {missing}")
    return tuple(selected)


def resolve_candidate_root(repository_root: Path, value: str) -> Path:
    """Resolve a candidate root confined to the repository checkout."""

    path = Path(value)
    if not path.is_absolute():
        path = repository_root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(
            f"candidate root escapes repository root: {resolved}"
        ) from error
    if not resolved.is_dir():
        raise ValueError(f"candidate root is not a directory: {resolved}")
    return resolved


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value, Sequence):
        raise TypeError(f"{context} must be a sequence")
    return value


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _non_negative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _positive_int(value: object, context: str) -> int:
    result = _non_negative_int(value, context)
    if result == 0:
        raise ValueError(f"{context} must be positive")
    return result


def _finite_non_negative(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be finite and non-negative")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _identity(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_node_semantic(node_id: str) -> str:
    match = _LAYER_NODE.search(node_id)
    if match is not None:
        return f"layer:{match.group('role')}"
    return node_id.rsplit(":", 1)[-1]


def _physical_key(spec: Mapping[str, object]) -> dict[str, object]:
    required = (
        "kind", "hardware_id", "variant", "dtype", "shape", "launch",
        "transfer",
    )
    missing = [name for name in required if name not in spec]
    if missing:
        raise ValueError(f"kernel spec is missing physical fields: {missing}")
    kind = _nonempty_string(spec["kind"], "kernel kind")
    launch = spec["launch"]
    transfer = spec["transfer"]
    if kind in _COMPUTE_KINDS:
        if launch is None:
            raise NotImplementedError(
                f"compute launch is unsupported for {kind}: launch is null")
        _mapping(launch, "kernel launch")
        if transfer is not None:
            raise ValueError(f"compute kernel {kind} cannot contain transfer")
        dtype = _nonempty_string(spec["dtype"], "kernel dtype")
        variant = _nonempty_string(spec["variant"], "kernel variant")
    else:
        if launch is not None:
            raise ValueError(f"transfer kernel {kind} cannot contain launch")
        _mapping(transfer, "kernel transfer")
        dtype = spec["dtype"]
        if dtype is not None:
            dtype = _nonempty_string(dtype, "kernel dtype")
        variant = spec["variant"]
        if variant is not None:
            variant = _nonempty_string(variant, "kernel variant")
    return {
        "kind": kind,
        "hardware_id": _nonempty_string(
            spec["hardware_id"], "kernel hardware_id"),
        "variant": variant,
        "dtype": dtype,
        "shape": dict(_mapping(spec["shape"], "kernel shape")),
        "launch": None if launch is None else dict(launch),
        "transfer": None if transfer is None else dict(transfer),
    }


def _workload_key(node: Mapping[str, object]) -> dict[str, object]:
    required = (
        "node_id", "kind", "pool_name", "shape", "dtype", "variant",
        "rank", "attributes",
    )
    missing = [name for name in required if name not in node]
    if missing:
        raise ValueError(f"workload node is missing fields: {missing}")
    node_id = _nonempty_string(node["node_id"], "workload node_id")
    dtype = node["dtype"]
    if dtype is not None:
        dtype = _nonempty_string(dtype, "workload dtype")
    variant = node["variant"]
    if variant is not None:
        variant = _nonempty_string(variant, "workload variant")
    return {
        "kind": _nonempty_string(node["kind"], "workload kind"),
        "variant": variant,
        "dtype": dtype,
        "shape": dict(_mapping(node["shape"], "workload shape")),
        "attributes": dict(_mapping(
            node["attributes"], "workload attributes")),
        "pool_name": _nonempty_string(
            node["pool_name"], "workload pool_name"),
        "rank": node["rank"],
        "source_node_semantic": _source_node_semantic(node_id),
    }


def _occurrence_counts(
    plan: Mapping[str, object],
    timeline: Mapping[str, object],
    call_ids: tuple[str, ...],
) -> dict[str, int]:
    repeated = "block_repeats" in plan and bool(plan["block_repeats"])
    if not repeated:
        return {call_id: 1 for call_id in call_ids}
    if "call_occurrence_counts" not in timeline:
        raise ValueError(
            "block-repeat timeline is missing call_occurrence_counts")
    rows = _sequence(
        timeline["call_occurrence_counts"], "call occurrence counts")
    counts: dict[str, int] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"call occurrence count {index}")
        if set(row) != {"call_id", "occurrence_count"}:
            raise ValueError("call occurrence count fields are invalid")
        call_id = _nonempty_string(row["call_id"], "occurrence call_id")
        if call_id in counts:
            raise ValueError(f"duplicate occurrence count for {call_id!r}")
        counts[call_id] = _positive_int(
            row["occurrence_count"], "occurrence_count")
    if set(counts) != set(call_ids):
        raise ValueError(
            "block-repeat occurrence counts must cover every call exactly")
    return counts


def _new_bucket(identity_value: Mapping[str, object]) -> dict[str, object]:
    return {
        "identity": dict(identity_value),
        "materialized_template_count": 0,
        "logical_occurrences": 0,
        "predicted_device_time_s": 0.0,
        "case_ids": set(),
        "step_count": 0,
    }


def _merge_bucket(
    destination: dict[str, object], source: Mapping[str, object], case_id: str,
) -> None:
    destination["materialized_template_count"] += source[
        "materialized_template_count"]
    destination["logical_occurrences"] += source["logical_occurrences"]
    destination["predicted_device_time_s"] += source[
        "predicted_device_time_s"]
    destination["step_count"] += source["step_count"]
    destination["case_ids"].add(case_id)


def _finalize_buckets(
    buckets: Mapping[str, Mapping[str, object]], identity_name: str,
) -> list[dict[str, object]]:
    rows = []
    for bucket_id, bucket in buckets.items():
        cases = bucket["case_ids"]
        if not isinstance(cases, set):
            raise TypeError("coverage bucket case_ids must be a set")
        rows.append({
            f"{identity_name}_id": bucket_id,
            identity_name: bucket["identity"],
            "materialized_template_count": bucket[
                "materialized_template_count"],
            "logical_occurrences": bucket["logical_occurrences"],
            "predicted_device_time_s": bucket["predicted_device_time_s"],
            "case_count": len(cases),
            "case_ids": sorted(cases),
            "step_count": bucket["step_count"],
        })
    return sorted(rows, key=lambda row: (
        -row["predicted_device_time_s"], row[f"{identity_name}_id"]
    ))


def _add_contribution_shares(
    rows: Sequence[dict[str, object]], total_s: float, prefix: str,
) -> None:
    cumulative_s = 0.0
    for row in rows:
        cumulative_s += row["predicted_device_time_s"]
        if total_s == 0.0:
            share = None
            cumulative_share = None
        else:
            share = row["predicted_device_time_s"] / total_s
            cumulative_share = min(1.0, cumulative_s / total_s)
        row[f"{prefix}_predicted_device_time_share"] = share
        row[f"{prefix}_cumulative_predicted_device_time_share"] = (
            cumulative_share)


def _consume_step(
    step_value: Mapping[str, object],
    case_id: str,
    physical_buckets: dict[str, dict[str, object]],
    op_buckets: dict[str, dict[str, object]],
) -> float:
    step = _mapping(step_value, "execution step")
    plan = _mapping(step["kernel_plan"], "kernel plan")
    calls = _sequence(plan["calls"], "kernel plan calls")
    call_rows = tuple(_mapping(call, "kernel call") for call in calls)
    call_ids = tuple(
        _nonempty_string(call["call_id"], "kernel call_id")
        for call in call_rows
    )
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("kernel plan contains duplicate call IDs")
    predictions_raw = _sequence(step["predictions"], "kernel predictions")
    predictions: dict[str, float] = {}
    for raw in predictions_raw:
        prediction = _mapping(raw, "kernel prediction")
        call_id = _nonempty_string(
            prediction["call_id"], "prediction call_id")
        if call_id in predictions:
            raise ValueError(f"duplicate prediction for {call_id!r}")
        predictions[call_id] = _finite_non_negative(
            prediction["t_total_s"], "prediction t_total_s")
    if set(predictions) != set(call_ids):
        raise ValueError("predictions must cover every kernel call exactly")
    timeline = _mapping(step["execution_timeline"], "execution timeline")
    intervals = _sequence(timeline["intervals"], "execution intervals")
    interval_ids = tuple(
        _nonempty_string(
            _mapping(value, "execution interval")["call_id"],
            "execution interval call_id",
        )
        for value in intervals
    )
    if len(interval_ids) != len(set(interval_ids)):
        raise ValueError("execution timeline contains duplicate call IDs")
    if set(interval_ids) != set(call_ids):
        raise ValueError("execution timeline must cover every call exactly")
    occurrences = _occurrence_counts(plan, timeline, call_ids)

    graph = _mapping(step["workload_graph"], "workload graph")
    nodes_raw = _sequence(graph["nodes"], "workload nodes")
    nodes: dict[str, Mapping[str, object]] = {}
    for raw in nodes_raw:
        node = _mapping(raw, "workload node")
        node_id = _nonempty_string(node["node_id"], "workload node_id")
        if node_id in nodes:
            raise ValueError(f"duplicate workload node {node_id!r}")
        nodes[node_id] = node
    lowering = _mapping(step["lowering"], "lowering")
    node_to_calls_raw = _sequence(
        lowering["node_to_call_ids"], "lowering node_to_call_ids")
    declared_pairs = set()
    for raw in node_to_calls_raw:
        row = _mapping(raw, "lowering node-to-call row")
        node_id = _nonempty_string(row["node_id"], "lowering node_id")
        if node_id not in nodes:
            raise ValueError("lowering references an unknown workload node")
        for call_id in _sequence(row["call_ids"], "lowering call_ids"):
            declared_pairs.add((node_id, call_id))

    step_physical: dict[str, dict[str, object]] = {}
    step_ops: dict[str, dict[str, object]] = {}
    contributions = []
    op_occurrences: dict[str, dict[str, set[int]]] = {}
    op_node_ids: dict[str, set[str]] = {}
    for call in call_rows:
        call_id = call["call_id"]
        source_ids = tuple(_sequence(
            call["source_node_ids"], "kernel source_node_ids"))
        for source_id in source_ids:
            if (source_id, call_id) not in declared_pairs:
                raise ValueError(
                    "kernel source_node_ids disagree with lowering mapping")
        spec = _mapping(call["kernel_spec"], "kernel spec")
        physical = _physical_key(spec)
        physical_id = _identity(physical)
        if physical_id not in step_physical:
            step_physical[physical_id] = _new_bucket(physical)
        physical_bucket = step_physical[physical_id]
        occurrence_count = occurrences[call_id]
        contribution = predictions[call_id] * occurrence_count
        physical_bucket["materialized_template_count"] += 1
        physical_bucket["logical_occurrences"] += occurrence_count
        physical_bucket["predicted_device_time_s"] += contribution
        contributions.append(contribution)

        if len(source_ids) == 1 and source_ids[0] in nodes:
            node_id = source_ids[0]
            workload = _workload_key(nodes[node_id])
            association = "exact_single_source_node"
        else:
            node_id = "|".join(sorted(str(value) for value in source_ids))
            workload = {
                "association": "ambiguous",
                "source_node_ids": sorted(str(value) for value in source_ids),
            }
            association = "ambiguous_source_node_ids"
        op_identity = {
            "association": association,
            "workload": workload,
        }
        op_id = _identity(op_identity)
        if op_id not in step_ops:
            step_ops[op_id] = _new_bucket(op_identity)
            step_ops[op_id]["physical_key_ids"] = set()
            op_occurrences[op_id] = {}
            op_node_ids[op_id] = set()
        op_bucket = step_ops[op_id]
        op_bucket["predicted_device_time_s"] += contribution
        op_bucket["physical_key_ids"].add(physical_id)
        if node_id not in op_occurrences[op_id]:
            op_occurrences[op_id][node_id] = set()
        op_occurrences[op_id][node_id].add(occurrence_count)
        op_node_ids[op_id].add(node_id)

    for physical_id, bucket in step_physical.items():
        bucket["step_count"] = 1
        if physical_id not in physical_buckets:
            physical_buckets[physical_id] = _new_bucket(bucket["identity"])
        _merge_bucket(physical_buckets[physical_id], bucket, case_id)
    for op_id, bucket in step_ops.items():
        logical_occurrences = 0
        for node_id, occurrence_values in op_occurrences[op_id].items():
            if len(occurrence_values) != 1:
                raise ValueError(
                    "leaf calls for one source OP disagree on occurrence "
                    f"count: {node_id}")
            logical_occurrences += next(iter(occurrence_values))
        bucket["logical_occurrences"] = logical_occurrences
        bucket["materialized_template_count"] = len(op_node_ids[op_id])
        bucket["step_count"] = 1
        if op_id not in op_buckets:
            op_buckets[op_id] = _new_bucket(bucket["identity"])
            op_buckets[op_id]["physical_key_ids"] = set()
        op_buckets[op_id]["physical_key_ids"].update(
            bucket["physical_key_ids"])
        _merge_bucket(op_buckets[op_id], bucket, case_id)
    step_total = math.fsum(contributions)
    physical_total = math.fsum(
        bucket["predicted_device_time_s"]
        for bucket in step_physical.values()
    )
    op_total = math.fsum(
        bucket["predicted_device_time_s"] for bucket in step_ops.values()
    )
    if not math.isclose(step_total, physical_total, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("step-to-physical contribution reconciliation failed")
    if not math.isclose(step_total, op_total, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("step-to-OP contribution reconciliation failed")
    return step_total


def reduce_execution_artifact(
    case_id: str, payload: Mapping[str, object],
) -> dict[str, object]:
    """Reduce one schema-v5/v7/v8 full artifact without retaining it."""

    artifact = _mapping(payload["execution_artifact"], "execution artifact")
    if artifact["schema_version"] not in _ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(
            "coverage requires execution artifact schema v5, v7, or v8")
    steps = _sequence(artifact["steps"], "execution artifact steps")
    physical: dict[str, dict[str, object]] = {}
    ops: dict[str, dict[str, object]] = {}
    totals = []
    for step in steps:
        totals.append(_consume_step(step, case_id, physical, ops))
    return _finalize_case(case_id, len(steps), physical, ops, totals)


def _finalize_case(
    case_id: str,
    step_count: int,
    physical: Mapping[str, Mapping[str, object]],
    ops: Mapping[str, Mapping[str, object]],
    step_totals: Sequence[float],
) -> dict[str, object]:
    physical_rows = _finalize_buckets(physical, "physical_key")
    op_rows = _finalize_buckets(ops, "op_key")
    for row in op_rows:
        bucket = ops[row["op_key_id"]]
        physical_ids = bucket["physical_key_ids"]
        if not isinstance(physical_ids, set):
            raise TypeError("OP physical_key_ids must be a set")
        row["physical_key_ids"] = sorted(physical_ids)
    step_total = math.fsum(step_totals)
    physical_total = math.fsum(
        row["predicted_device_time_s"] for row in physical_rows)
    op_total = math.fsum(
        row["predicted_device_time_s"] for row in op_rows)
    reconciled = (
        math.isclose(step_total, physical_total, rel_tol=1e-12, abs_tol=1e-15)
        and math.isclose(step_total, op_total, rel_tol=1e-12, abs_tol=1e-15)
    )
    if not reconciled:
        raise ValueError("case contribution reconciliation failed")
    _add_contribution_shares(physical_rows, step_total, "case")
    _add_contribution_shares(op_rows, step_total, "case")
    return {
        "cell_id": case_id,
        "status": "complete",
        "failure": None,
        "step_count": step_count,
        "physical_kernels": physical_rows,
        "logical_ops": op_rows,
        "reconciliation": {
            "step_sum_s": step_total,
            "physical_key_sum_s": physical_total,
            "logical_op_sum_s": op_total,
            "passed": True,
        },
    }


def explicit_failure(
    case_id: str,
    stage: str,
    error: BaseException,
    *,
    completed_steps: int = 0,
) -> dict[str, object]:
    """Return a visible case failure; no partial contribution is published."""

    return {
        "cell_id": case_id,
        "status": "failed",
        "failure": {
            "stage": _nonempty_string(stage, "failure stage"),
            "exception_type": type(error).__name__,
            "message": str(error),
        },
        "step_count": completed_steps,
        "physical_kernels": [],
        "logical_ops": [],
        "reconciliation": {
            "step_sum_s": None,
            "physical_key_sum_s": None,
            "logical_op_sum_s": None,
            "passed": False,
        },
    }


def _captured_schedule_steps(case: object) -> Iterable[ScheduleStep]:
    if case.scheduler_steps is None:
        raise ValueError("captured coverage requires schema-v3 scheduler_steps")
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
        finished = tuple(
            f"request-{_non_negative_int(value, 'finished request_idx'):012d}"
            for value in _sequence(
                step["finished_request_indexes"], "finished request indexes")
        )
        yield ScheduleStep(
            step_id=f"step-{step_index:012d}",
            pool_name="main",
            requests=tuple(requests),
            finished_request_ids=finished,
            preempted_request_ids=(),
            attention_kv_layout=None,
        )


def _alignment_gate(case: object) -> dict[str, object]:
    if case.scheduler_steps is None:
        raise ValueError("alignment gate requires schema-v3 scheduler_steps")
    samples_by_index = {
        sample.source_execution_order: sample
        for sample in case.alignment_steps.values()
        if sample is not None
    }
    reconstructed = {}
    for index, schedule in enumerate(_captured_schedule_steps(case)):
        if index in samples_by_index:
            reconstructed[index] = schedule
    phases = []
    for phase in ("prefill", "decode", "mixed"):
        sample = case.alignment_steps[phase]
        if sample is None:
            phases.append({"phase": phase, "status": "not_observed"})
            continue
        index = sample.source_execution_order
        if index not in reconstructed:
            phases.append({
                "phase": phase,
                "status": "mismatched",
                "reason": "source_execution_order exceeds scheduler trace",
            })
            continue
        expected = asdict(compact_step_alignment.deserialize_schedule_step(
            sample.scheduler))
        actual = asdict(reconstructed[index])
        expected_sha = _identity(expected)
        actual_sha = _identity(actual)
        if expected == actual:
            status = "matched"
            differing_fields: list[str] = []
        else:
            status = "mismatched"
            differing_fields = sorted(
                name for name in set(expected) | set(actual)
                if name not in expected or name not in actual
                or expected[name] != actual[name]
            )
        phases.append({
            "phase": phase,
            "status": status,
            "source_execution_order": index,
            "expected_schedule_sha256": expected_sha,
            "reconstructed_schedule_sha256": actual_sha,
            "differing_fields": differing_fields,
        })
    observed = [row for row in phases if row["status"] != "not_observed"]
    return {
        "status": (
            "passed" if observed
            and all(row["status"] == "matched" for row in observed)
            else "failed"
        ),
        "phases": phases,
    }


def _captured_case(
    repository_root: Path,
    config: compact.Phase1Config,
    cell: compact.Phase1Cell,
    case: object,
    execution_mode: str,
) -> dict[str, object]:
    alignment = _alignment_gate(case)
    model_path = repository_root / cell.model.simulation_model_path
    hardware_path = repository_root / cell.hardware.hardware_config_path
    model = load_model_config(str(model_path))
    hardware, prediction, deployment = (
        compact_step_alignment.load_case_setup(
            config, model, hardware_path))

    routing = compact._expected_expert_routing(repository_root, cell)
    runtime_inputs = RuntimeInputs(
        expert_routing=routing,
        attention_kv_layout=prediction["attention_kv_layout"],
        attention_kv_distribution=prediction["attention_kv_distribution"],
        graph_execution=cell.hardware.graph_execution,
    )
    execution = ServingExecutionContext(
        model,
        deployment,
        {hardware.hardware_id: hardware},
        prediction["launch_policy"],
        prediction["prediction_method"],
        runtime_inputs,
        execution_mode=execution_mode,
    )
    physical: dict[str, dict[str, object]] = {}
    ops: dict[str, dict[str, object]] = {}
    totals = []
    completed_steps = 0
    for schedule in _captured_schedule_steps(case):
        components = execution.execute_step(schedule)
        step_manifest = {
            "step_id": schedule.step_id,
            **execution_step_components_manifest(
                components.workload,
                components.kernel_plan,
                components.predictions,
                components.execution_timeline,
                origin_time_s=0.0,
            ),
        }
        totals.append(_consume_step(
            step_manifest, cell.cell_id, physical, ops))
        completed_steps += 1
        del step_manifest, components, schedule
    result = _finalize_case(
        cell.cell_id, completed_steps, physical, ops, totals)
    result["alignment_reconstruction_gate"] = alignment
    if model.num_experts:
        routing_status = "routing_profile_predicted_not_actual_shape"
    else:
        routing_status = "not_applicable_dense_model"
    result["routing_shape_provenance"] = routing_status
    result["hard_gate_pass"] = (
        alignment["status"] == "passed"
        and routing_status != "routing_profile_predicted_not_actual_shape"
    )
    return result


def _predicted_case(
    repository_root: Path,
    config: compact.Phase1Config,
    cell: compact.Phase1Cell,
    case: object,
    execution_mode: str,
    simulation_runner_fn: Callable[[Sequence[str]], Mapping[str, object]],
) -> dict[str, object]:
    argv = compact.build_simulation_argv(
        repository_root,
        config,
        cell,
        case,
        execution_mode=execution_mode,
        serving_timing_mode="device_only",
    )
    argv.extend(("--step-retention", "full"))
    payload = simulation_runner_fn(argv)
    try:
        result = reduce_execution_artifact(cell.cell_id, payload)
    finally:
        del payload
        gc.collect()
    result["alignment_reconstruction_gate"] = {
        "status": "not_applicable_predicted_scheduler_control",
        "phases": [],
    }
    result["routing_shape_provenance"] = (
        "simulation_runtime_input_not_capture_truth")
    result["hard_gate_pass"] = False
    result["simulation_argv"] = argv
    return result


def classify_timing_source(
    family: str, entry: Mapping[str, object],
    hardware_id: str = "metax_c500",
) -> dict[str, object]:
    """Classify canonical timing without using duration as evidence."""

    op_info = _mapping(entry["op_info"], "canonical op_info")
    evaluation = _mapping(
        op_info["evaluation"], "canonical op evaluation")
    timing = _mapping(evaluation["timing"], "canonical op timing")
    source = timing["source"]
    granularity = timing["granularity"]
    if hardware_id == "metax_c500" and source == "mcTracer":
        if granularity != "kernel":
            raise ValueError("mcTracer canonical timing must be kernel-level")
        classification = "legacy_mctracer_fallback"
        warning = None
    elif hardware_id == "metax_c500" and source == "device_event_op_median":
        if granularity != "op":
            raise ValueError("device Event canonical timing must be OP-level")
        if family == "vector":
            classification = "event_warning"
            warning = "event_stability_unverified"
        elif family in ("gemm", "flash_attn", "fused_moe"):
            classification = "strict_event"
            warning = None
        else:
            raise ValueError(f"unknown C500 dataset family: {family!r}")
    else:
        classification = "canonical_dataset_timing"
        warning = None
    return {
        "classification": classification,
        "granularity": granularity,
        "source": source,
        "warning": warning,
        "op_type": op_info["Type"],
    }


def load_canonical_timing_index(
    repository_root: str | Path,
    hardware_id: str = "metax_c500",
) -> dict[str, tuple[dict[str, object], ...]]:
    root = Path(repository_root).resolve()
    index = {}
    for family, filename in _FAMILY_FILENAMES.items():
        path = root / "kernel_dataset" / hardware_id / filename
        payload = _mapping(
            load_kernel_dataset_document(path), f"{family} dataset")
        rows = []
        for case_id, raw_entry in payload.items():
            entry = _mapping(raw_entry, f"canonical case {case_id}")
            op_info = _mapping(entry["op_info"], "canonical op_info")
            rows.append({
                "case_id": case_id,
                "family": family,
                "op_type": op_info["Type"],
                "attributes": dict(_mapping(
                    op_info["Attr"], "canonical attributes")),
                "timing": classify_timing_source(
                    family, entry, hardware_id),
            })
        index[family] = tuple(rows)
    return index


def _candidate_query(
    op_key: Mapping[str, object],
) -> tuple[str | None, str | None, Mapping[str, object] | None]:
    if op_key["association"] != "exact_single_source_node":
        return None, "ambiguous_source_node_ids", None
    workload = _mapping(op_key["workload"], "OP workload key")
    dtype = workload["dtype"]
    kind = workload["kind"]
    variant = workload["variant"]
    shape = _mapping(workload["shape"], "OP workload shape")
    attributes = _mapping(workload["attributes"], "OP workload attributes")
    if kind in ("linear", "lm_head", "router"):
        query = {
            "B": shape["b"], "M": shape["m"], "K": shape["k"],
            "N": shape["n"], "dtype": dtype,
        }
        return "gemm", None, query
    if dtype != "bf16":
        return None, "canonical_non_gemm_dtype_is_not_explicit", None
    if kind == "flash_attn":
        query = {
            "B": shape["batch"], "Q_LEN": shape["q_len"],
            "KV_LEN": shape["kv_len"], "HQ": shape["num_query_heads"],
            "HKV": shape["num_kv_heads"], "D": shape["head_dim"],
            "Q_LENS": list(attributes["q_lens"]),
            "KV_LENS": list(attributes["kv_lens"]),
            "kv_layout": attributes["kv_layout"],
            "kv_dist": attributes["kv_dist"], "causal": attributes["causal"],
        }
        return "flash_attn", None, query
    if kind == "fused_moe":
        query = {
            "TOK": shape["tokens"], "E": shape["experts"],
            "TOPK": shape["topk"], "H": shape["hidden"],
            "INT": shape["intermediate"],
            "tok_per_expert": list(attributes["tokens_per_expert"]),
        }
        return "fused_moe", None, query
    vector_types = {
        "norm": "NORM", "rotary": "ROTARY", "activation": "ACTIVATION",
        "elementwise": "ELEMENTWISE", "kv_update": "KV_CACHE",
    }
    if kind not in vector_types:
        return None, "kernel_family_not_in_canonical_c500_dataset", None
    op_names = {
        "norm": variant,
        "rotary": "rope",
        "activation": variant,
        "elementwise": "elementwise",
        "kv_update": "kvcache",
    }
    query = {"op": op_names[kind]}
    if kind == "elementwise":
        query["op2"] = variant
    if "tok" in shape:
        query["tok"] = shape["tok"]
    elif "rows" in shape:
        query["tok"] = shape["rows"]
    elif "seq_len" in shape:
        query["tok"] = shape["seq_len"]
    if "hidden" in shape:
        query["hidden"] = shape["hidden"]
    if "element_count" in shape:
        query["element_count"] = shape["element_count"]
    elif "elements" in shape:
        query["element_count"] = shape["elements"]
    if "num_query_heads" in shape:
        query["num_heads"] = shape["num_query_heads"]
    if "num_kv_heads" in shape:
        query["num_kv_heads"] = shape["num_kv_heads"]
    if "head_dim" in shape:
        query["head_dim"] = shape["head_dim"]
    if "intermediate" in shape:
        query["intermediate"] = shape["intermediate"]
    query["dtype_bytes"] = 2
    if "element_count" not in query and "tok" in query and "hidden" in query:
        query["element_count"] = query["tok"] * query["hidden"]
    query["Type"] = vector_types[kind]
    return "vector", None, query


def _annotate_golden(
    op_rows: Sequence[dict[str, object]],
    canonical_index: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    expert_parallel_size: int,
) -> None:
    for row in op_rows:
        op_key = _mapping(row["op_key"], "OP key")
        family, reason, query = _candidate_query(op_key)
        if family is None or query is None:
            row["canonical_mapping"] = {
                "status": "unmapped", "reason": reason, "matches": [],
                "timing": None,
            }
            continue
        if family == "fused_moe" and expert_parallel_size > 1:
            row["canonical_mapping"] = {
                "status": "unmapped",
                "reason": "ep2_rank_local_moe_is_not_ep1_full_op_golden",
                "matches": [], "timing": None,
            }
            continue
        matches = []
        for candidate in canonical_index[family]:
            attributes = candidate["attributes"]
            expected_type = None
            if "Type" in query:
                expected_type = query["Type"]
            if expected_type is not None and candidate["op_type"] != expected_type:
                continue
            compared = tuple(
                (name, value) for name, value in query.items()
                if name != "Type"
            )
            if all(name in attributes and attributes[name] == value
                   for name, value in compared):
                matches.append(candidate)
        if not matches:
            mapping = {
                "status": "missing", "reason": "no_exact_semantic_match",
                "matches": [], "timing": None,
            }
        elif len(matches) > 1:
            mapping = {
                "status": "ambiguous",
                "reason": "multiple_canonical_semantic_matches",
                "matches": sorted(candidate["case_id"] for candidate in matches),
                "timing": None,
            }
        else:
            candidate = matches[0]
            mapping = {
                "status": "exact",
                "reason": None,
                "matches": [candidate["case_id"]],
                "timing": candidate["timing"],
            }
        row["canonical_mapping"] = mapping


def _merge_case_rows(
    cases: Sequence[Mapping[str, object]],
    row_name: str,
    identity_name: str,
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    id_name = f"{identity_name}_id"
    for case in cases:
        if case["status"] != "complete":
            continue
        case_id = case["cell_id"]
        for row in case[row_name]:
            row_id = row[id_name]
            if row_id not in merged:
                merged[row_id] = _new_bucket(row[identity_name])
                if identity_name == "op_key":
                    merged[row_id]["physical_key_ids"] = set()
            source = {
                "materialized_template_count": row[
                    "materialized_template_count"],
                "logical_occurrences": row["logical_occurrences"],
                "predicted_device_time_s": row["predicted_device_time_s"],
                "step_count": row["step_count"],
            }
            _merge_bucket(merged[row_id], source, case_id)
            if identity_name == "op_key":
                merged[row_id]["physical_key_ids"].update(
                    row["physical_key_ids"])
    rows = _finalize_buckets(merged, identity_name)
    if identity_name == "op_key":
        for row in rows:
            identifiers = merged[row[id_name]]["physical_key_ids"]
            row["physical_key_ids"] = sorted(identifiers)
    return rows


def build_report(
    case_results: Sequence[dict[str, object]],
    canonical_index: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    input_mode: str,
    expert_parallel_size_by_cell: Mapping[str, int],
    hardware_id: str = "metax_c500",
) -> dict[str, object]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unknown coverage input mode: {input_mode!r}")
    physical_rows = _merge_case_rows(
        case_results, "physical_kernels", "physical_key")
    op_rows = _merge_case_rows(case_results, "logical_ops", "op_key")
    for case in case_results:
        if case["status"] == "complete":
            _annotate_golden(
                case["logical_ops"], canonical_index,
                expert_parallel_size=expert_parallel_size_by_cell[
                    case["cell_id"]],
            )
    for row in op_rows:
        contributing_ep_sizes = {
            expert_parallel_size_by_cell[cell_id]
            for cell_id in row["case_ids"]
        }
        _annotate_golden(
            [row], canonical_index,
            expert_parallel_size=max(contributing_ep_sizes),
        )
    completed = [case for case in case_results if case["status"] == "complete"]
    failed = [case for case in case_results if case["status"] == "failed"]
    case_total = math.fsum(
        case["reconciliation"]["step_sum_s"] for case in completed)
    physical_total = math.fsum(
        row["predicted_device_time_s"] for row in physical_rows)
    op_total = math.fsum(
        row["predicted_device_time_s"] for row in op_rows)
    reconciliation_passed = (
        math.isclose(case_total, physical_total, rel_tol=1e-12, abs_tol=1e-15)
        and math.isclose(case_total, op_total, rel_tol=1e-12, abs_tol=1e-15)
    )
    if not reconciliation_passed:
        raise ValueError("global contribution reconciliation failed")
    _add_contribution_shares(physical_rows, case_total, "global")
    _add_contribution_shares(op_rows, case_total, "global")
    hard_gate_pass = (
        not failed and bool(completed)
        and all(case["hard_gate_pass"] for case in completed)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "hardware_id": hardware_id,
        "input_mode": input_mode,
        "hard_gate_pass": hard_gate_pass,
        "case_counts": {
            "requested": len(case_results),
            "complete": len(completed),
            "failed": len(failed),
        },
        "reconciliation": {
            "case_sum_s": case_total,
            "physical_key_sum_s": physical_total,
            "logical_op_sum_s": op_total,
            "passed": True,
        },
        "cases": list(case_results),
        "physical_kernels": physical_rows,
        "logical_ops": op_rows,
    }


def _case_path(
    repository_root: Path,
    config: compact.Phase1Config,
    cell: compact.Phase1Cell,
    candidate_root: Path | None,
) -> Path:
    canonical_path = compact.compact_case_path(repository_root, config, cell)
    if candidate_root is None:
        return canonical_path
    dataset_root = (repository_root / config.dataset_root).resolve()
    relative = canonical_path.relative_to(dataset_root)
    selected = (candidate_root.resolve() / relative).resolve()
    if candidate_root.resolve() not in selected.parents:
        raise ValueError("coverage candidate path escapes candidate root")
    return selected


def run_coverage_census(
    repository_root: str | Path,
    selections: Sequence[tuple[compact.Phase1Config, compact.Phase1Cell]],
    *,
    input_mode: str = "captured_scheduler_replay",
    execution_mode: str = "accelerated",
    candidate_root: str | Path | None = None,
    jobs: int = 1,
    simulation_runner_fn: Callable[[Sequence[str]], Mapping[str, object]] = (
        simulation_runner.run_cli_args),
    compact_case_loader: Callable[[Path], object],
) -> dict[str, object]:
    root = Path(repository_root).resolve()
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unknown coverage input mode: {input_mode!r}")
    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise TypeError("coverage jobs must be an integer")
    if jobs < 1:
        raise ValueError("coverage jobs must be positive")
    selected_root = None if candidate_root is None else Path(candidate_root)
    hardware_ids = {
        cell.hardware.hardware_id for _, cell in selections
    }
    if len(hardware_ids) != 1:
        raise ValueError(
            "coverage census requires exactly one hardware, got "
            f"{sorted(hardware_ids)}")
    hardware_id = next(iter(hardware_ids))
    canonical_index = load_canonical_timing_index(root, hardware_id)
    ep_sizes = {}
    for config, cell in selections:
        ep_sizes[cell.cell_id] = config.prediction.values[
            "expert_parallel_size"]

    def run_one(
        selection: tuple[compact.Phase1Config, compact.Phase1Cell],
    ) -> dict[str, object]:
        config, cell = selection
        path = _case_path(root, config, cell, selected_root)
        case = compact_case_loader(path)
        argv = compact.build_simulation_argv(
            root, config, cell, case,
            execution_mode=execution_mode,
            serving_timing_mode="device_only",
        )
        if "--host-launch-profile" in argv:
            raise AssertionError("device-only coverage received Host timing")
        if input_mode == "captured_scheduler_replay":
            result = _captured_case(root, config, cell, case, execution_mode)
            result["simulation_argv_contract"] = [
                *argv, "--step-retention", "full",
            ]
        elif input_mode == "predicted_scheduler_control":
            result = _predicted_case(
                root, config, cell, case, execution_mode,
                simulation_runner_fn,
            )
        else:
            raise ValueError(f"unsupported coverage input mode: {input_mode!r}")
        gc.collect()
        return result

    if jobs == 1:
        results = [run_one(selection) for selection in selections]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(run_one, selections))
    return build_report(
        results,
        canonical_index,
        input_mode=input_mode,
        expert_parallel_size_by_cell=ep_sizes,
        hardware_id=hardware_id,
    )


def write_report(report: Mapping[str, object], output_dir: str | Path) -> Path:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    json_path = destination / "kernel_coverage.json"
    csv_path = destination / "kernel_coverage.csv"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = (
        "row_type", "identity_id", "kind", "variant", "dtype",
        "materialized_template_count", "logical_occurrences",
        "predicted_device_time_s", "case_count", "case_ids",
        "contribution_share", "cumulative_contribution_share",
        "canonical_status", "canonical_case_ids", "timing_classification",
        "timing_source", "warning", "identity_json",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["physical_kernels"]:
            identity = row["physical_key"]
            writer.writerow({
                "row_type": "physical_kernel",
                "identity_id": row["physical_key_id"],
                "kind": identity["kind"],
                "variant": identity["variant"],
                "dtype": identity["dtype"],
                "materialized_template_count": row[
                    "materialized_template_count"],
                "logical_occurrences": row["logical_occurrences"],
                "predicted_device_time_s": row["predicted_device_time_s"],
                "case_count": row["case_count"],
                "case_ids": ";".join(row["case_ids"]),
                "contribution_share": row[
                    "global_predicted_device_time_share"],
                "cumulative_contribution_share": row[
                    "global_cumulative_predicted_device_time_share"],
                "canonical_status": "not_applicable_physical_leaf",
                "canonical_case_ids": "",
                "timing_classification": "",
                "timing_source": "",
                "warning": "",
                "identity_json": _canonical_json(identity),
            })
        for row in report["logical_ops"]:
            identity = row["op_key"]
            mapping = row["canonical_mapping"]
            timing = mapping["timing"]
            if identity["association"] == "exact_single_source_node":
                workload = identity["workload"]
                kind = workload["kind"]
                variant = workload["variant"]
                dtype = workload["dtype"]
            elif identity["association"] == "ambiguous_source_node_ids":
                kind = "ambiguous"
                variant = ""
                dtype = ""
            else:
                raise ValueError(
                    f"unsupported logical association: {identity['association']!r}"
                )
            writer.writerow({
                "row_type": "logical_op",
                "identity_id": row["op_key_id"],
                "kind": kind,
                "variant": variant,
                "dtype": dtype,
                "materialized_template_count": row[
                    "materialized_template_count"],
                "logical_occurrences": row["logical_occurrences"],
                "predicted_device_time_s": row["predicted_device_time_s"],
                "case_count": row["case_count"],
                "case_ids": ";".join(row["case_ids"]),
                "contribution_share": row[
                    "global_predicted_device_time_share"],
                "cumulative_contribution_share": row[
                    "global_cumulative_predicted_device_time_share"],
                "canonical_status": mapping["status"],
                "canonical_case_ids": ";".join(mapping["matches"]),
                "timing_classification": (
                    "" if timing is None else timing["classification"]),
                "timing_source": "" if timing is None else timing["source"],
                "warning": "" if timing is None or timing["warning"] is None
                else timing["warning"],
                "identity_json": _canonical_json(identity),
            })
    return destination
