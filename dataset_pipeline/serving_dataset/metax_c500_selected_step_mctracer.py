"""Correlate one validated C500 selected-step injection with mcTracer tasks."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence

from dataset_pipeline.conversion.mctracer_chrome import (
    iter_mctracer_trace_events,
)

from dataset_pipeline.serving_dataset.interval_ops import (
    clipped_union_ns,
    merged_clipped_intervals,
    subtract_intervals,
)
from dataset_pipeline.serving_dataset.json_shape import (
    compact_mapping,
    compact_sequence,
    file_sha256,
    interval_union_ns,
)


_MAX_CLOCK_RESIDUAL_NS = 1_000_000
_KERNEL_LAUNCH_APIS = frozenset(("mcLaunchKernel", "mcModuleLaunchKernel"))


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a non-empty string")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{context} must be finite")
    return normalized


def _exact_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )




def _load_json(path: Path) -> Mapping[str, object]:
    return compact_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            rows.append(
                compact_mapping(json.loads(line), f"{path}:{line_number}")
            )
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _filename_pid(path: Path, prefix: str) -> int:
    expected_prefix = f"{prefix}."
    expected_suffix = ".jsonl"
    if not path.name.startswith(expected_prefix) or not path.name.endswith(
        expected_suffix
    ):
        raise ValueError(f"invalid C500 correlation filename: {path.name}")
    text = path.name[len(expected_prefix):-len(expected_suffix)]
    if not text.isdecimal():
        raise ValueError(f"invalid C500 correlation PID in {path.name}")
    return int(text)


def _percentile(values: Sequence[int], percentile: int) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("clock residual percentile requires samples")
    index = math.ceil(percentile / 100 * len(ordered)) - 1
    return ordered[max(index, 0)]


def _context_records(
    path: Path, expected_step_ids: set[str]
) -> tuple[int, int, dict[str, Mapping[str, object]]]:
    records = _load_jsonl(path)
    pid = None
    rank = None
    by_step: dict[str, list[Mapping[str, object]]] = {}
    for index, record in enumerate(records):
        context = f"{path}:record[{index}]"
        _exact_keys(
            record,
            {
                "schema_version", "source", "event", "process_pid",
                "thread_id", "host_monotonic_ns", "context",
            },
            context,
        )
        if record["schema_version"] != 1 or record["source"] != (
            "midlayer_c500_scheduler_step_context"
        ):
            raise ValueError(f"{context} has unsupported identity")
        record_pid = _integer(record["process_pid"], f"{context}.process_pid", minimum=1)
        details = compact_mapping(record["context"], f"{context}.context")
        _exact_keys(
            details,
            {"trace_step_id", "worker_rank", "context_ordinal"},
            f"{context}.context",
        )
        record_rank = _integer(details["worker_rank"], f"{context}.worker_rank")
        if pid is None:
            pid = record_pid
            rank = record_rank
        if record_pid != pid or record_rank != rank:
            raise ValueError(f"{path} mixes process or rank identity")
        step_id = _string(details["trace_step_id"], f"{context}.trace_step_id")
        if step_id in expected_step_ids:
            if step_id not in by_step:
                by_step[step_id] = []
            by_step[step_id].append(record)
    if pid is None or rank is None:
        raise ValueError(f"{path} has no C500 context records")
    if _filename_pid(path, "graph_context") != pid:
        raise ValueError(f"{path} filename PID differs from its records")
    result = {}
    for step_id in expected_step_ids:
        step_records = by_step[step_id] if step_id in by_step else []
        if len(step_records) != 2:
            raise ValueError(
                f"C500 context {step_id} must have exactly begin/end records"
            )
        begin, end = step_records
        if begin["event"] != "begin" or end["event"] != "end":
            raise ValueError(f"C500 context {step_id} is not a begin/end pair")
        if begin["context"] != end["context"]:
            raise ValueError(f"C500 context {step_id} identity changed")
        begin_ns = _integer(begin["host_monotonic_ns"], "context begin", minimum=1)
        end_ns = _integer(end["host_monotonic_ns"], "context end", minimum=1)
        if end_ns <= begin_ns:
            raise ValueError(f"C500 context {step_id} has an invalid interval")
        result[step_id] = {
            "begin_ns": begin_ns,
            "end_ns": end_ns,
            "context": begin["context"],
        }
    return pid, rank, result


def _launch_records(
    path: Path,
    pid: int,
    contexts: Mapping[str, Mapping[str, object]],
    graph_step_ids: set[str] | None = None,
) -> tuple[list[Mapping[str, object]], dict[str, Mapping[str, object]]]:
    if graph_step_ids is None:
        graph_step_ids = set(contexts)
    if not graph_step_ids <= set(contexts):
        raise ValueError("C500 Graph step IDs escape the selected contexts")
    launches = _load_jsonl(path)
    selected: dict[str, list[Mapping[str, object]]] = {
        step_id: [] for step_id in contexts
    }
    for index, launch in enumerate(launches):
        context = f"{path}:record[{index}]"
        _exact_keys(
            launch,
            {
                "schema_version", "source", "process_pid", "thread_id",
                "launch_ordinal", "host_monotonic_start_ns",
                "host_monotonic_end_ns", "graph_exec", "stream", "result",
                "context",
            },
            context,
        )
        if launch["schema_version"] != 1 or launch["source"] != (
            "midlayer_c500_mcgraphlaunch_interposer"
        ):
            raise ValueError(f"{context} has unsupported identity")
        if _integer(launch["process_pid"], f"{context}.process_pid", minimum=1) != pid:
            raise ValueError(f"{path} process PID differs from its context log")
        if _integer(launch["launch_ordinal"], f"{context}.launch_ordinal") != index:
            raise ValueError(f"{path} launch ordinals are not contiguous")
        start_ns = _integer(
            launch["host_monotonic_start_ns"], f"{context}.start", minimum=1
        )
        end_ns = _integer(
            launch["host_monotonic_end_ns"], f"{context}.end", minimum=1
        )
        if end_ns <= start_ns or launch["result"] != 0:
            raise ValueError(f"{context} failed or has an invalid interval")
        launch_context = launch["context"]
        if launch_context is None:
            continue
        launch_context = compact_mapping(launch_context, f"{context}.context")
        step_id = _string(
            launch_context["trace_step_id"], f"{context}.trace_step_id"
        )
        if step_id in selected:
            expected = contexts[step_id]
            if launch_context != expected["context"]:
                raise ValueError(f"C500 launch context differs for {step_id}")
            if not (expected["begin_ns"] <= start_ns < end_ns <= expected["end_ns"]):
                raise ValueError(f"C500 launch escapes context interval for {step_id}")
            selected[step_id].append(launch)
    result = {}
    for step_id, matches in selected.items():
        expected_count = 1 if step_id in graph_step_ids else 0
        if len(matches) != expected_count:
            raise ValueError(
                f"C500 {step_id} must have exactly {expected_count} "
                "interposed Graph launches"
            )
        if matches:
            result[step_id] = matches[0]
    return launches, result


def _raw_rank_trace(
    path: Path,
    pid: int,
    rank: int,
    interposed_launches: Sequence[Mapping[str, object]],
    selected_launches: Mapping[str, Mapping[str, object]],
    required_device_step_ids: set[str] | None = None,
    eager_contexts: Mapping[str, Mapping[str, object]] | None = None,
    allow_unanchored_device_union: bool = False,
) -> dict[str, object]:
    if eager_contexts is None:
        eager_contexts = {}
    if required_device_step_ids is None:
        required_device_step_ids = set(selected_launches) | set(eager_contexts)
    selected_step_ids = set(selected_launches) | set(eager_contexts)
    if set(selected_launches) & set(eager_contexts):
        raise ValueError("C500 step cannot be both Graph and eager")
    if not required_device_step_ids <= selected_step_ids:
        raise ValueError("C500 required device steps are not selected runtime steps")
    selected_ordinals = {
        _integer(launch["launch_ordinal"], "selected launch ordinal"): step_id
        for step_id, launch in selected_launches.items()
    }
    raw_launches = []
    target_co_ids: dict[int, str] = {}
    intervals_by_step: dict[str, list[tuple[int, int]]] = {
        step_id: [] for step_id in selected_step_ids
    }
    task_types_by_step: dict[str, Counter[str]] = {
        step_id: Counter() for step_id in selected_step_ids
    }
    task_groups_by_step: dict[str, dict[str, dict[str, object]]] = {
        step_id: {} for step_id in selected_step_ids
    }
    launch_api_intervals_by_step: dict[str, list[tuple[int, int]]] = {
        step_id: [] for step_id in selected_step_ids
    }
    non_launch_api_intervals_by_step: dict[str, list[tuple[int, int]]] = {
        step_id: [] for step_id in selected_step_ids
    }
    for event_index, event in iter_mctracer_trace_events(path):
        if "args" not in event or not isinstance(event["args"], dict):
            continue
        args = compact_mapping(event["args"], f"raw event[{event_index}].args")
        if "name" in event and event["name"] == "mcGraphLaunch":
            if "process_id" not in args or args["process_id"] != pid:
                continue
            ordinal = len(raw_launches)
            raw_launches.append((event_index, event))
            if ordinal in selected_ordinals:
                co_id = _integer(
                    args["co_id"], f"raw Graph launch[{ordinal}].co_id", minimum=1
                )
                if co_id in target_co_ids:
                    raise ValueError("C500 selected Graph correlation ID is reused")
                target_co_ids[co_id] = selected_ordinals[ordinal]
    if len(raw_launches) != len(interposed_launches):
        raise ValueError(
            "C500 raw/interposer Graph launch count mismatch: "
            f"raw={len(raw_launches)}, interposed={len(interposed_launches)}"
        )

    offsets_by_ordinal = []
    raw_co_ids = set()
    for ordinal, ((event_index, raw), interposed) in enumerate(
        zip(raw_launches, interposed_launches, strict=True)
    ):
        args = compact_mapping(raw["args"], f"raw Graph launch[{ordinal}].args")
        if raw["ph"] != "X" or args["ret"] != 0:
            raise ValueError(f"raw Graph launch[{ordinal}] failed or is not complete")
        co_id = _integer(args["co_id"], f"raw Graph launch[{ordinal}].co_id", minimum=1)
        if co_id in raw_co_ids:
            raise ValueError(f"raw Graph launch correlation ID is duplicated: {co_id}")
        raw_co_ids.add(co_id)
        raw_start_ns = _integer(raw["ts"], f"raw Graph launch[{ordinal}].ts", minimum=1)
        host_start_ns = _integer(
            interposed["host_monotonic_start_ns"],
            f"interposed Graph launch[{ordinal}].start",
            minimum=1,
        )
        offsets_by_ordinal.append(raw_start_ns - host_start_ns)
        if _integer(interposed["launch_ordinal"], "interposed launch ordinal") != ordinal:
            raise ValueError("C500 interposed launch order changed")

    formal_ordinals = sorted(selected_ordinals)
    if allow_unanchored_device_union:
        if len(formal_ordinals) != 1:
            raise ValueError(
                "C500 unanchored device-union analysis requires one selected "
                "Graph launch"
            )
        local_anchor_ordinals = [formal_ordinals[0]]
        offset_ns = offsets_by_ordinal[formal_ordinals[0]]
        clock_mapping = {
            "method": "single_graph_launch_start_alignment_unvalidated",
            "anchor_ordinals": local_anchor_ordinals,
            "raw_minus_host_monotonic_ns": offset_ns,
            "maximum_residual_ns": None,
            "p99_residual_ns": None,
            "host_clock_comparable": False,
        }
    else:
        if not formal_ordinals:
            raise ValueError(
                "C500 selected interval requires a local Graph clock anchor"
            )
        if len(formal_ordinals) > 1 and any(
            right != left + 1
            for left, right in zip(formal_ordinals, formal_ordinals[1:])
        ):
            raise ValueError("C500 selected formal Graph launches must be adjacent")
        if formal_ordinals[0] == 0:
            raise ValueError(
                "C500 selected Graph launch requires a preceding local clock anchor"
            )
        local_anchor_ordinals = [formal_ordinals[0] - 1, *formal_ordinals]
        if len(formal_ordinals) == 1 and formal_ordinals[0] + 1 < len(raw_launches):
            local_anchor_ordinals.append(formal_ordinals[0] + 1)
        local_offsets = [
            offsets_by_ordinal[ordinal] for ordinal in local_anchor_ordinals
        ]
        offset_ns = round(statistics.median(local_offsets))
        residuals = [abs(value - offset_ns) for value in local_offsets]
        if max(residuals) > _MAX_CLOCK_RESIDUAL_NS:
            raise ValueError(
                "C500 local mcTracer/host clock mapping residual exceeds 1 ms: "
                f"rank={rank}, max_ns={max(residuals)}"
            )
        clock_mapping = {
            "method": "median_local_graph_launch_start_minus_interposer_start",
            "anchor_ordinals": local_anchor_ordinals,
            "raw_minus_host_monotonic_ns": offset_ns,
            "maximum_residual_ns": max(residuals),
            "p99_residual_ns": _percentile(residuals, 99),
        }

    raw_contexts = {
        step_id: (
            _integer(context["begin_ns"], f"{step_id} context begin", minimum=1)
            + offset_ns,
            _integer(context["end_ns"], f"{step_id} context end", minimum=1)
            + offset_ns,
        )
        for step_id, context in eager_contexts.items()
    }
    for event_index, event in iter_mctracer_trace_events(path):
        if "args" not in event or not isinstance(event["args"], dict):
            continue
        args = compact_mapping(event["args"], f"raw event[{event_index}].args")
        if (
            event.get("ph") != "X"
            or "device_id" in args
            or args.get("process_id") != pid
        ):
            continue
        start_ns = _integer(event["ts"], f"raw host event[{event_index}].ts")
        duration_ns = _integer(
            event["dur"], f"raw host event[{event_index}].dur", minimum=1
        )
        matching_steps = [
            step_id
            for step_id, (begin_ns, end_ns) in raw_contexts.items()
            if begin_ns <= start_ns and start_ns + duration_ns <= end_ns
        ]
        if len(matching_steps) > 1:
            raise ValueError("C500 eager Host API belongs to multiple step contexts")
        if not matching_steps:
            continue
        step_id = matching_steps[0]
        projected_interval = (start_ns - offset_ns, start_ns + duration_ns - offset_ns)
        if args.get("link") is True:
            launch_api_intervals_by_step[step_id].append(projected_interval)
        else:
            non_launch_api_intervals_by_step[step_id].append(projected_interval)
        if args.get("link") is not True or "co_id" not in args:
            continue
        co_id = _integer(args["co_id"], f"raw host event[{event_index}].co_id")
        if co_id in target_co_ids and target_co_ids[co_id] != step_id:
            raise ValueError("C500 correlation ID crosses selected step contexts")
        target_co_ids[co_id] = step_id

    # mcTracer does not guarantee that correlated device records follow their
    # CPU API record in file order. Resolve all selected co_ids first, then
    # stream the raw file again and retain only those device intervals.
    for event_index, event in iter_mctracer_trace_events(path):
        if "args" not in event or not isinstance(event["args"], dict):
            continue
        args = compact_mapping(event["args"], f"raw event[{event_index}].args")
        if "co_id" not in args or args["co_id"] not in target_co_ids:
            continue
        if event.get("ph") != "X" or "device_id" not in args:
            continue
        step_id = target_co_ids[args["co_id"]]
        event_device = _integer(
            args["device_id"], f"raw device event[{event_index}].device_id"
        )
        if event_device != rank:
            raise ValueError(
                f"C500 rank/device mismatch: rank={rank}, device={event_device}"
            )
        start_ns = _integer(event["ts"], f"raw device event[{event_index}].ts")
        duration_ns = _integer(
            event["dur"], f"raw device event[{event_index}].dur", minimum=1
        )
        intervals_by_step[step_id].append((start_ns, start_ns + duration_ns))
        task_type = "kernel" if "grid" in args else "memory"
        task_types_by_step[step_id][task_type] += 1
        name = _string(event["name"], f"raw device event[{event_index}].name")
        launch = {
            "grid": args["grid"] if "grid" in args else None,
            "block": args["block"] if "block" in args else None,
            "dynamic_shared_bytes": (
                compact_mapping(args["mem"], f"raw device event[{event_index}].mem")[
                    "dynamic_shared"
                ]
                if "mem" in args
                else None
            ),
            "bytes": args["bytes"] if "bytes" in args else None,
            "copy_kind": args["copy_kind"] if "copy_kind" in args else None,
            "source_kind": args["src_kind"] if "src_kind" in args else None,
            "destination_kind": args["dst_kind"] if "dst_kind" in args else None,
        }
        signature = {"task_type": task_type, "name": name, "launch": launch}
        signature_key = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        groups = task_groups_by_step[step_id]
        if signature_key not in groups:
            groups[signature_key] = {
                **signature,
                "count": 0,
                "additive_duration_ns": 0,
            }
        groups[signature_key]["count"] += 1
        groups[signature_key]["additive_duration_ns"] += duration_ns
    steps = {}
    for step_id, intervals in intervals_by_step.items():
        if step_id in required_device_step_ids and not intervals:
            raise ValueError(f"C500 selected Graph has no device tasks: {step_id}")
        projected = [(start - offset_ns, end - offset_ns) for start, end in intervals]
        steps[step_id] = {
            "task_count": len(projected),
            "task_types": dict(sorted(task_types_by_step[step_id].items())),
            "task_groups": sorted(
                task_groups_by_step[step_id].values(),
                key=lambda group: (
                    -group["additive_duration_ns"],
                    group["task_type"],
                    group["name"],
                    json.dumps(group["launch"], sort_keys=True),
                ),
            ),
            "device_union_ns": interval_union_ns(projected),
            "intervals": projected,
            "launch_api_intervals": launch_api_intervals_by_step[step_id],
            "non_launch_api_intervals": non_launch_api_intervals_by_step[step_id],
        }
    return {
        "rank": rank,
        "process_pid": pid,
        "trace_path": str(path.resolve()),
        "trace_sha256": file_sha256(path),
        "raw_graph_launch_count": len(raw_launches),
        "clock_mapping": clock_mapping,
        "steps": steps,
    }


def _kernel_launch_records(
    path: Path,
    pid: int,
    target_step_id: str,
    target_context: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    records = _load_jsonl(path)
    by_ordinal = {}
    selected = []
    for index, record in enumerate(records):
        context = f"{path}:record[{index}]"
        _exact_keys(
            record,
            {
                "schema_version", "source", "process_pid", "thread_id",
                "launch_ordinal", "api", "host_monotonic_start_ns",
                "host_monotonic_end_ns", "function", "grid", "block",
                "dynamic_shared_memory_bytes", "stream", "result", "context",
            },
            context,
        )
        if record["schema_version"] != 1 or record["source"] != (
            "midlayer_c500_mckernellaunch_interposer"
        ):
            raise ValueError(f"{context} has unsupported identity")
        if _integer(record["process_pid"], f"{context}.process_pid", minimum=1) != pid:
            raise ValueError(f"{path} process PID differs from its context log")
        ordinal = _integer(record["launch_ordinal"], f"{context}.launch_ordinal")
        if ordinal in by_ordinal:
            raise ValueError(f"{path} duplicates kernel launch ordinal {ordinal}")
        if record["api"] not in _KERNEL_LAUNCH_APIS:
            raise ValueError(f"{context} has unsupported kernel-launch API")
        start_ns = _integer(
            record["host_monotonic_start_ns"], f"{context}.start", minimum=1
        )
        end_ns = _integer(
            record["host_monotonic_end_ns"], f"{context}.end", minimum=1
        )
        if end_ns <= start_ns or record["result"] != 0:
            raise ValueError(f"{context} failed or has an invalid interval")
        by_ordinal[ordinal] = record
        launch_context = record["context"]
        if launch_context is None:
            continue
        launch_context = compact_mapping(launch_context, f"{context}.context")
        if launch_context["trace_step_id"] != target_step_id:
            continue
        if launch_context != target_context["context"]:
            raise ValueError("C500 eager kernel context identity changed")
        if not (
            target_context["begin_ns"] <= start_ns < end_ns
            <= target_context["end_ns"]
        ):
            raise ValueError("C500 eager kernel launch escapes target context")
        selected.append(record)
    expected_ordinals = list(range(len(records)))
    if sorted(by_ordinal) != expected_ordinals:
        raise ValueError(f"{path} kernel launch ordinals are not contiguous")
    if not selected:
        raise ValueError("C500 eager target contains no interposed kernel launches")
    return [by_ordinal[value] for value in expected_ordinals], selected


def _raw_eager_rank_trace(
    path: Path,
    pid: int,
    rank: int,
    interposed_launches: Sequence[Mapping[str, object]],
    selected_launches: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    raw_launches = []
    for event_index, event in iter_mctracer_trace_events(path):
        if event["name"] not in _KERNEL_LAUNCH_APIS:
            continue
        args = compact_mapping(event["args"], f"raw launch[{event_index}].args")
        if args["process_id"] != pid:
            continue
        raw_launches.append((event_index, event))
    if len(raw_launches) != len(interposed_launches):
        raise ValueError(
            "C500 raw/interposer kernel launch count mismatch: "
            f"raw={len(raw_launches)}, interposed={len(interposed_launches)}"
        )

    selected_ordinals = {
        _integer(value["launch_ordinal"], "selected eager launch ordinal")
        for value in selected_launches
    }
    target_co_ids = {}
    raw_co_ids = set()
    offsets = []
    for ordinal, ((event_index, raw), interposed) in enumerate(
        zip(raw_launches, interposed_launches, strict=True)
    ):
        if interposed["launch_ordinal"] != ordinal:
            raise ValueError("C500 interposed eager launch order changed")
        if raw["ph"] != "X":
            raise ValueError(f"raw eager launch[{ordinal}] is not complete")
        # mcTracer and the interposer can serialize unrelated concurrent
        # initialization calls with different API order.  They must still
        # have the same complete count and ordinal domain above, while the
        # selected context itself remains exact: it is the only launch range
        # used to derive a device-time result.
        if (
            ordinal in selected_ordinals
            and raw["name"] != interposed["api"]
        ):
            raise ValueError(f"raw eager launch[{ordinal}] API identity differs")
        args = compact_mapping(raw["args"], f"raw launch[{event_index}].args")
        if args["ret"] != 0:
            raise ValueError(f"raw eager launch[{ordinal}] failed")
        co_id = _integer(args["co_id"], f"raw launch[{ordinal}].co_id", minimum=1)
        if co_id in raw_co_ids:
            raise ValueError("C500 eager kernel correlation ID is reused")
        raw_co_ids.add(co_id)
        if ordinal in selected_ordinals:
            target_co_ids[co_id] = interposed
            raw_start_ns = _integer(
                raw["ts"], f"raw launch[{ordinal}].ts", minimum=1
            )
            offsets.append(
                raw_start_ns
                - _integer(
                    interposed["host_monotonic_start_ns"],
                    f"interposed launch[{ordinal}].start",
                    minimum=1,
                )
            )
    if len(target_co_ids) != len(selected_launches):
        raise ValueError("C500 eager target launch correlation is incomplete")

    offset_ns = round(statistics.median(offsets))
    residuals = [abs(value - offset_ns) for value in offsets]
    if max(residuals) > _MAX_CLOCK_RESIDUAL_NS:
        raise ValueError(
            "C500 eager mcTracer/host clock mapping residual exceeds 1 ms: "
            f"rank={rank}, max_ns={max(residuals)}"
        )
    observed_co_ids = set()
    intervals = []
    groups = {}
    kernels = []
    device_id = None
    for event_index, event in iter_mctracer_trace_events(path):
        if "args" not in event or not isinstance(event["args"], dict):
            continue
        args = compact_mapping(event["args"], f"raw event[{event_index}].args")
        if "co_id" not in args or args["co_id"] not in target_co_ids:
            continue
        if event["ph"] != "X" or "device_id" not in args or "grid" not in args:
            continue
        co_id = args["co_id"]
        if co_id in observed_co_ids:
            raise ValueError(f"C500 eager launch {co_id} has multiple device kernels")
        observed_co_ids.add(co_id)
        event_device = _integer(
            args["device_id"], f"raw kernel[{event_index}].device_id"
        )
        if device_id is None:
            device_id = event_device
        if event_device != device_id:
            raise ValueError(
                f"C500 rank {rank} spans multiple devices: "
                f"{device_id}, {event_device}"
            )
        interposed = target_co_ids[co_id]
        grid = compact_mapping(args["grid"], f"raw kernel[{event_index}].grid")
        block = compact_mapping(args["block"], f"raw kernel[{event_index}].block")
        raw_grid = [grid[axis] for axis in ("x", "y", "z")]
        raw_block = [block[axis] for axis in ("x", "y", "z")]
        if raw_grid != interposed["grid"] or raw_block != interposed["block"]:
            raise ValueError("C500 eager interposer/mcTracer launch geometry differs")
        start_ns = _integer(event["ts"], f"raw kernel[{event_index}].ts", minimum=1)
        duration_ns = _integer(
            event["dur"], f"raw kernel[{event_index}].dur", minimum=1
        )
        projected = (start_ns - offset_ns, start_ns + duration_ns - offset_ns)
        intervals.append(projected)
        name = _string(event["name"], f"raw kernel[{event_index}].name")
        launch = {
            "api": interposed["api"],
            "grid": raw_grid,
            "block": raw_block,
            "dynamic_shared_bytes": interposed["dynamic_shared_memory_bytes"],
        }
        signature = {"name": name, "launch": launch}
        signature_key = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        if signature_key not in groups:
            groups[signature_key] = {
                **signature,
                "count": 0,
                "additive_duration_ns": 0,
                "duration_samples_ns": [],
            }
        groups[signature_key]["count"] += 1
        groups[signature_key]["additive_duration_ns"] += duration_ns
        groups[signature_key]["duration_samples_ns"].append(duration_ns)
        kernels.append({
            "launch_ordinal": interposed["launch_ordinal"],
            "co_id": co_id,
            "name": name,
            "launch": launch,
            "duration_ns": duration_ns,
            "interval": list(projected),
        })
    if observed_co_ids != set(target_co_ids):
        raise ValueError(
            "C500 eager target has missing device kernels: "
            f"{sorted(set(target_co_ids) - observed_co_ids)}"
        )
    return {
        "rank": rank,
        "device_id": device_id,
        "process_pid": pid,
        "trace_path": str(path.resolve()),
        "trace_sha256": file_sha256(path),
        "raw_kernel_launch_count": len(raw_launches),
        "target_kernel_count": len(kernels),
        "device_kernel_union_ns": interval_union_ns(intervals),
        "clock_mapping": {
            "method": "median_target_kernel_launch_start_minus_interposer_start",
            "raw_minus_host_monotonic_ns": offset_ns,
            "maximum_residual_ns": max(residuals),
            "p99_residual_ns": _percentile(residuals, 99),
        },
        "kernel_groups": sorted(
            groups.values(),
            key=lambda value: (-value["additive_duration_ns"], value["name"]),
        ),
        "kernels": sorted(kernels, key=lambda value: value["launch_ordinal"]),
    }


def analyze_c500_selected_step_capture(capture_dir: Path) -> dict[str, object]:
    """Return a fail-closed selected/following-step mcTracer timeline."""

    capture_dir = capture_dir.resolve(strict=True)
    validation_path = capture_dir / "validation.json"
    replay_path = capture_dir / "injection_replay.json"
    scheduler_path = capture_dir / "scheduler_steps.raw.jsonl"
    clock_path = Path(f"{scheduler_path}.clock.json")
    validation = _load_json(validation_path)
    replay = _load_json(replay_path)
    if validation["kind"] != "selected_step_injection_validation":
        raise ValueError("C500 analysis requires selected-step validation")
    if validation["structural_fingerprint_match"] is not True:
        raise ValueError("INVALID EXPERIMENT — structural fingerprint mismatch")
    if replay["schema_version"] != 5:
        raise ValueError("C500 analysis requires replay schema 5")
    if replay["forward_duration_observation_consumed"] is not False:
        raise ValueError("C500 analysis refuses observation-fed Forward")
    mapping = compact_mapping(validation["step_id_mapping"], "step ID mapping")
    target_step_id = _string(mapping["injected_step_id"], "injected step ID")
    rows = _load_jsonl(scheduler_path)
    target_indexes = [
        index for index, row in enumerate(rows) if row["step_id"] == target_step_id
    ]
    if len(target_indexes) != 1:
        raise ValueError("C500 target step must occur exactly once")
    target_index = target_indexes[0]
    if target_index == 0 or target_index + 1 >= len(rows):
        raise ValueError("C500 target requires previous and following steps")
    following_step_id = _string(
        rows[target_index + 1]["step_id"], "following step ID"
    )
    step_ids = (target_step_id, following_step_id)
    expected_step_ids = set(step_ids)
    rows_by_step = {row["step_id"]: row for row in rows}
    graph_step_ids = set()
    eager_step_ids = set()
    for step_id in step_ids:
        runtime_mode = _string(
            rows_by_step[step_id]["runtime_cudagraph_mode"],
            f"{step_id} runtime Graph mode",
        )
        if runtime_mode == "FULL":
            graph_step_ids.add(step_id)
        elif runtime_mode == "NONE":
            eager_step_ids.add(step_id)
        else:
            raise ValueError(
                "C500 selected-step mcTracer analysis does not yet support "
                f"runtime Graph mode {runtime_mode!r}"
            )

    replay_rows = [
        compact_mapping(value, "replay step comparison")
        for value in compact_sequence(replay["step_comparison"], "replay comparisons")
    ]
    replay_target = [row for row in replay_rows if row["step_id"] == target_step_id]
    if len(replay_target) != 1:
        raise ValueError("C500 replay target step must occur exactly once")
    fingerprint = compact_mapping(replay_target[0]["step_fingerprint"], "step fingerprint")
    graph = compact_mapping(fingerprint["graph_execution"], "step graph execution")
    expected_ranks = {
        _integer(value, "step graph rank")
        for value in compact_sequence(graph["rank_ids"], "step graph ranks")
    }

    context_paths = sorted(capture_dir.glob("graph_context.*.jsonl"))
    launch_paths = sorted(capture_dir.glob("graph_launch.*.jsonl"))
    if len(context_paths) != len(expected_ranks) or len(launch_paths) != len(expected_ranks):
        raise ValueError("C500 correlation file count differs from expected ranks")
    rank_inputs = {}
    launch_by_pid = {
        _filename_pid(path, "graph_launch"): path
        for path in launch_paths
    }
    for context_path in context_paths:
        pid, rank, contexts = _context_records(context_path, expected_step_ids)
        if rank in rank_inputs:
            raise ValueError(f"duplicate C500 worker rank {rank}")
        if pid not in launch_by_pid:
            raise ValueError(f"missing C500 launch log for PID {pid}")
        launches, selected_launches = _launch_records(
            launch_by_pid[pid], pid, contexts, graph_step_ids
        )
        trace_path = capture_dir / "trace" / f"tracer_out-{pid}.json"
        if not trace_path.is_file():
            raise FileNotFoundError(f"missing C500 mcTracer worker trace: {trace_path}")
        rank_inputs[rank] = _raw_rank_trace(
            trace_path,
            pid,
            rank,
            launches,
            selected_launches,
            required_device_step_ids={target_step_id},
            eager_contexts={
                step_id: contexts[step_id] for step_id in eager_step_ids
            },
        )
    if set(rank_inputs) != expected_ranks:
        raise ValueError(
            f"C500 observed ranks differ: {sorted(rank_inputs)} != {sorted(expected_ranks)}"
        )

    clock = _load_json(clock_path)
    if clock["clock_domain"] != "shared_host_time.perf_counter_ns":
        raise ValueError("C500 Scheduler trace clock domain is unsupported")
    origin_ns = _integer(
        clock["trace_origin_monotonic_ns"], "Scheduler trace origin", minimum=1
    )
    results = []
    for step_id in (target_step_id,):
        indexes = [index for index, row in enumerate(rows) if row["step_id"] == step_id]
        if len(indexes) != 1 or indexes[0] == 0:
            raise ValueError(f"C500 step boundary is missing or ambiguous: {step_id}")
        index = indexes[0]
        row = rows[index]
        previous = rows[index - 1]
        service_start_ns = origin_ns + max(
            _integer(row["schedule_end_ns"], "schedule end"),
            _integer(previous["update_start_ns"], "previous update start"),
        )
        service_end_ns = origin_ns + _integer(
            row["update_start_ns"], "update start"
        )
        if service_end_ns <= service_start_ns:
            raise ValueError(f"C500 {step_id} serialized service is not positive")
        device_intervals = [
            interval
            for rank in sorted(rank_inputs)
            for interval in rank_inputs[rank]["steps"][step_id]["intervals"]
        ]
        device_union_ns = interval_union_ns(device_intervals)
        exposed_ns = clipped_union_ns(
            device_intervals, service_start_ns, service_end_ns
        )
        service_ns = service_end_ns - service_start_ns
        service_intervals = [(service_start_ns, service_end_ns)]
        device_busy_intervals = merged_clipped_intervals(
            device_intervals, service_start_ns, service_end_ns
        )
        device_idle_intervals = subtract_intervals(
            service_intervals, device_busy_intervals
        )
        launch_api_intervals = [
            interval
            for rank in sorted(rank_inputs)
            for interval in rank_inputs[rank]["steps"][step_id][
                "launch_api_intervals"
            ]
        ]
        non_launch_api_intervals = [
            interval
            for rank in sorted(rank_inputs)
            for interval in rank_inputs[rank]["steps"][step_id][
                "non_launch_api_intervals"
            ]
        ]
        launch_api_exposed = merged_clipped_intervals(
            launch_api_intervals, service_start_ns, service_end_ns
        )
        launch_api_exposed = [
            interval
            for idle_start, idle_end in device_idle_intervals
            for interval in merged_clipped_intervals(
                launch_api_exposed, idle_start, idle_end
            )
        ]
        idle_after_launch = subtract_intervals(
            device_idle_intervals, launch_api_exposed
        )
        non_launch_api_exposed = [
            interval
            for idle_start, idle_end in idle_after_launch
            for interval in merged_clipped_intervals(
                non_launch_api_intervals, idle_start, idle_end
            )
        ]
        idle_outside_api = subtract_intervals(
            idle_after_launch, non_launch_api_exposed
        )
        results.append({
            "step_id": step_id,
            "serialized_service_ms": service_ns / 1_000_000,
            "device_task_union_ms": device_union_ns / 1_000_000,
            "exposed_device_task_union_ms": exposed_ns / 1_000_000,
            "uncovered_serialized_service_ms": (service_ns - exposed_ns) / 1_000_000,
            "uncovered_breakdown_ms": {
                "launch_api": interval_union_ns(launch_api_exposed) / 1_000_000,
                "non_launch_api": interval_union_ns(non_launch_api_exposed) / 1_000_000,
                "outside_visible_runtime_api": interval_union_ns(idle_outside_api) / 1_000_000,
            },
            "runtime_inter_task_gap_ms": None,
            "per_rank": [
                {
                    "rank": rank,
                    "process_pid": rank_inputs[rank]["process_pid"],
                    "task_count": rank_inputs[rank]["steps"][step_id]["task_count"],
                    "task_types": rank_inputs[rank]["steps"][step_id]["task_types"],
                    "task_groups": rank_inputs[rank]["steps"][step_id][
                        "task_groups"
                    ],
                    "device_union_ms": (
                        rank_inputs[rank]["steps"][step_id]["device_union_ns"]
                        / 1_000_000
                    ),
                    "launch_api_union_ms": interval_union_ns(
                        rank_inputs[rank]["steps"][step_id][
                            "launch_api_intervals"
                        ]
                    ) / 1_000_000,
                    "non_launch_api_union_ms": interval_union_ns(
                        rank_inputs[rank]["steps"][step_id][
                            "non_launch_api_intervals"
                        ]
                    ) / 1_000_000,
                }
                for rank in sorted(rank_inputs)
            ],
        })
    return {
        "schema_version": 2,
        "kind": "c500_selected_step_mctracer_timeline",
        "comparison_status": validation["comparison_status"],
        "content_identity": validation["content_identity"],
        "cell_id": validation["cell_id"],
        "step_id_mapping": dict(mapping),
        "following_clock_anchor_step_id": following_step_id,
        "clock_domain": "shared_host_time.perf_counter_ns",
        "source_sha256": {
            "validation": file_sha256(validation_path),
            "injection_replay": file_sha256(replay_path),
            "scheduler_trace": file_sha256(scheduler_path),
            "scheduler_clock": file_sha256(clock_path),
        },
        "rank_traces": [
            {
                key: value
                for key, value in rank_inputs[rank].items()
                if key != "steps"
            }
            for rank in sorted(rank_inputs)
        ],
        "steps": results,
    }


def analyze_c500_unanchored_graph_device_union(
    capture_dir: Path,
) -> dict[str, object]:
    """Expose a one-Graph capture's per-rank device union without timing it.

    A single Graph launch has no residual-checked mapping from the mcTracer
    clock to the Scheduler's host clock.  Its device intervals can still
    describe each rank's own union duration, but cannot support a serialized
    service, uncovered-runtime, or cross-rank-union comparison.
    """

    capture_dir = capture_dir.resolve(strict=True)
    validation_path = capture_dir / "validation.json"
    replay_path = capture_dir / "injection_replay.json"
    scheduler_path = capture_dir / "scheduler_steps.raw.jsonl"
    validation = _load_json(validation_path)
    replay = _load_json(replay_path)
    if validation["kind"] != "selected_step_injection_validation":
        raise ValueError("C500 analysis requires selected-step validation")
    if validation["structural_fingerprint_match"] is not True:
        raise ValueError("INVALID EXPERIMENT — structural fingerprint mismatch")
    if replay["schema_version"] != 5:
        raise ValueError("C500 analysis requires replay schema 5")
    if replay["forward_duration_observation_consumed"] is not False:
        raise ValueError("C500 analysis refuses observation-fed Forward")
    mapping = compact_mapping(validation["step_id_mapping"], "step ID mapping")
    target_step_id = _string(mapping["injected_step_id"], "injected step ID")
    rows = _load_jsonl(scheduler_path)
    target_rows = [row for row in rows if row["step_id"] == target_step_id]
    if len(target_rows) != 1:
        raise ValueError("C500 target step must occur exactly once")

    replay_rows = [
        compact_mapping(value, "replay step comparison")
        for value in compact_sequence(replay["step_comparison"], "replay comparisons")
    ]
    replay_target = [row for row in replay_rows if row["step_id"] == target_step_id]
    if len(replay_target) != 1:
        raise ValueError("C500 replay target step must occur exactly once")
    fingerprint = compact_mapping(replay_target[0]["step_fingerprint"], "step fingerprint")
    graph = compact_mapping(fingerprint["graph_execution"], "step graph execution")
    if graph["runtime_mode"] != "FULL":
        raise ValueError("C500 unanchored analysis requires a FULL Graph target")
    expected_ranks = {
        _integer(value, "step graph rank")
        for value in compact_sequence(graph["rank_ids"], "step graph ranks")
    }

    context_paths = sorted(capture_dir.glob("graph_context.*.jsonl"))
    launch_paths = sorted(capture_dir.glob("graph_launch.*.jsonl"))
    if len(context_paths) != len(expected_ranks) or len(launch_paths) != len(
        expected_ranks
    ):
        raise ValueError("C500 correlation file count differs from expected ranks")
    launch_by_pid = {
        _filename_pid(path, "graph_launch"): path for path in launch_paths
    }
    rank_inputs = {}
    for context_path in context_paths:
        pid, rank, contexts = _context_records(context_path, {target_step_id})
        if rank in rank_inputs:
            raise ValueError(f"duplicate C500 worker rank {rank}")
        if pid not in launch_by_pid:
            raise ValueError(f"missing C500 launch log for PID {pid}")
        launches, selected_launches = _launch_records(
            launch_by_pid[pid], pid, contexts
        )
        trace_path = capture_dir / "trace" / f"tracer_out-{pid}.json"
        if not trace_path.is_file():
            raise FileNotFoundError(f"missing C500 mcTracer worker trace: {trace_path}")
        rank_inputs[rank] = _raw_rank_trace(
            trace_path,
            pid,
            rank,
            launches,
            selected_launches,
            required_device_step_ids={target_step_id},
            allow_unanchored_device_union=True,
        )
    if set(rank_inputs) != expected_ranks:
        raise ValueError(
            f"C500 observed ranks differ: {sorted(rank_inputs)} != "
            f"{sorted(expected_ranks)}"
        )

    return {
        "schema_version": 1,
        "kind": "c500_selected_step_unanchored_graph_device_union",
        "comparison_status": "unsupported_no_residual_checked_host_clock_anchor",
        "capture_validation_status": validation["comparison_status"],
        "content_identity": validation["content_identity"],
        "cell_id": validation["cell_id"],
        "step_id_mapping": dict(mapping),
        "target_step_id": target_step_id,
        "device_union_contract": (
            "per_rank_only; serialized service, uncovered runtime, and "
            "cross-rank device union are unsupported"
        ),
        "unsupported": [
            "no_previous_local_Graph_clock_anchor",
            "no_following_Graph_clock_anchor",
            "no_residual_checked_mcTracer_to_host_clock_mapping",
        ],
        "source_sha256": {
            "validation": file_sha256(validation_path),
            "injection_replay": file_sha256(replay_path),
            "scheduler_trace": file_sha256(scheduler_path),
        },
        "per_rank": [
            {
                "rank": rank,
                "process_pid": rank_inputs[rank]["process_pid"],
                "task_count": rank_inputs[rank]["steps"][target_step_id][
                    "task_count"
                ],
                "task_types": rank_inputs[rank]["steps"][target_step_id][
                    "task_types"
                ],
                "task_groups": rank_inputs[rank]["steps"][target_step_id][
                    "task_groups"
                ],
                "device_union_ms": (
                    rank_inputs[rank]["steps"][target_step_id]["device_union_ns"]
                    / 1_000_000
                ),
                "clock_mapping": rank_inputs[rank]["clock_mapping"],
            }
            for rank in sorted(rank_inputs)
        ],
    }


def analyze_c500_independent_eager_capture(
    capture_dir: Path,
) -> dict[str, object]:
    """Correlate eager kernel launches for one frozen selected-step workload."""

    capture_dir = capture_dir.resolve(strict=True)
    validation_path = capture_dir / "validation.json"
    replay_path = capture_dir / "injection_replay.json"
    scheduler_path = capture_dir / "scheduler_steps.raw.jsonl"
    validation = _load_json(validation_path)
    replay = _load_json(replay_path)
    if validation["kind"] != "selected_step_injection_validation":
        raise ValueError("C500 eager analysis requires selected-step validation")
    if validation["structural_fingerprint_match"] is not True:
        raise ValueError("INVALID EXPERIMENT — eager structural fingerprint mismatch")
    if validation["execution_boundary"] != "independent_eager":
        raise ValueError("C500 eager analysis requires an independent eager replay")
    if replay["schema_version"] != 5 or replay["analysis_kind"] != (
        "independent_eager_selected_step_injection_replay"
    ):
        raise ValueError("C500 eager analysis requires eager replay schema 5")
    if replay["forward_duration_observation_consumed"] is not False:
        raise ValueError("C500 eager analysis refuses observation-fed Forward")
    mapping = compact_mapping(validation["step_id_mapping"], "step ID mapping")
    target_step_id = _string(mapping["injected_step_id"], "injected step ID")
    rows = _load_jsonl(scheduler_path)
    targets = [row for row in rows if row["step_id"] == target_step_id]
    if len(targets) != 1:
        raise ValueError("C500 eager target step must occur exactly once")
    target_row = targets[0]
    if target_row["configured_cudagraph_mode"] != "NONE" or target_row[
        "runtime_cudagraph_mode"
    ] != "NONE":
        raise ValueError("C500 eager target unexpectedly used Graph execution")
    if target_row["capture_bucket_tokens"] is not None or target_row[
        "capture_limit_tokens"
    ] is not None:
        raise ValueError("C500 eager target unexpectedly has a Graph capture bucket")
    if target_row["padding_tokens"] != 0:
        raise ValueError("C500 eager target unexpectedly contains Graph padding")

    replay_rows = [
        compact_mapping(value, "replay step comparison")
        for value in compact_sequence(replay["step_comparison"], "replay comparisons")
    ]
    replay_targets = [row for row in replay_rows if row["step_id"] == target_step_id]
    if len(replay_targets) != 1:
        raise ValueError("C500 eager replay target step must occur exactly once")
    fingerprint = compact_mapping(replay_targets[0]["step_fingerprint"], "step fingerprint")
    graph = compact_mapping(fingerprint["graph_execution"], "step graph execution")
    expected_ranks = {
        _integer(value, "step graph rank")
        for value in compact_sequence(graph["rank_ids"], "step graph ranks")
    }

    context_paths = sorted(capture_dir.glob("graph_context.*.jsonl"))
    kernel_paths = sorted(capture_dir.glob("kernel_launch.*.jsonl"))
    if len(context_paths) != len(expected_ranks) or len(kernel_paths) != len(
        expected_ranks
    ):
        raise ValueError("C500 eager correlation file count differs from expected ranks")
    kernel_by_pid = {
        _filename_pid(path, "kernel_launch"): path for path in kernel_paths
    }
    rank_inputs = {}
    for context_path in context_paths:
        pid, rank, contexts = _context_records(context_path, {target_step_id})
        if rank in rank_inputs:
            raise ValueError(f"duplicate C500 eager worker rank {rank}")
        if pid not in kernel_by_pid:
            raise ValueError(f"missing C500 eager kernel launch log for PID {pid}")
        launches, selected_launches = _kernel_launch_records(
            kernel_by_pid[pid], pid, target_step_id, contexts[target_step_id]
        )
        trace_path = capture_dir / "trace" / f"tracer_out-{pid}.json"
        if not trace_path.is_file():
            raise FileNotFoundError(
                f"missing C500 eager mcTracer worker trace: {trace_path}"
            )
        rank_inputs[rank] = _raw_eager_rank_trace(
            trace_path, pid, rank, launches, selected_launches
        )
    if set(rank_inputs) != expected_ranks:
        raise ValueError(
            f"C500 eager observed ranks differ: {sorted(rank_inputs)} != "
            f"{sorted(expected_ranks)}"
        )

    critical_path_ns = max(
        rank_inputs[rank]["device_kernel_union_ns"] for rank in rank_inputs
    )
    return {
        "schema_version": 1,
        "kind": "c500_independent_eager_selected_step_mctracer",
        "comparison_status": "independent_eager_not_comparable_to_canonical_graph_time",
        "content_identity": validation["content_identity"],
        "cell_id": validation["cell_id"],
        "step_id_mapping": dict(mapping),
        "graph_execution": dict(graph),
        "target_kernel_critical_path_ms": critical_path_ns / 1_000_000,
        "source_sha256": {
            "validation": file_sha256(validation_path),
            "injection_replay": file_sha256(replay_path),
            "scheduler_trace": file_sha256(scheduler_path),
        },
        "per_rank": [rank_inputs[rank] for rank in sorted(rank_inputs)],
    }


def analyze_c500_standalone_prefill_capture(
    capture_dir: Path,
) -> dict[str, object]:
    """Correlate a canonical-mode eager standalone prefill without host union.

    A C500 above-capture-limit prefill has no Graph launch boundary, so a
    rank-local kernel union is the only valid device timing result here.  It
    deliberately does not claim a host-service comparison or cross-rank clock
    union.
    """

    capture_dir = capture_dir.resolve(strict=True)
    validation_path = capture_dir / "validation.json"
    replay_path = capture_dir / "injection_replay.json"
    scheduler_path = capture_dir / "scheduler_steps.raw.jsonl"
    validation = _load_json(validation_path)
    replay = _load_json(replay_path)
    if validation["kind"] != "selected_step_injection_validation":
        raise ValueError("C500 prefill analysis requires selected-step validation")
    if validation["structural_fingerprint_match"] is not True:
        raise ValueError("INVALID EXPERIMENT — structural fingerprint mismatch")
    if validation["execution_boundary"] != "canonical_graph_mode":
        raise ValueError("C500 prefill analysis requires canonical Graph mode")
    if replay["schema_version"] != 5:
        raise ValueError("C500 prefill analysis requires replay schema 5")
    if replay["forward_duration_observation_consumed"] is not False:
        raise ValueError("C500 prefill analysis refuses observation-fed Forward")
    mapping = compact_mapping(validation["step_id_mapping"], "step ID mapping")
    target_step_id = _string(mapping["injected_step_id"], "injected step ID")
    rows = _load_jsonl(scheduler_path)
    target_rows = [row for row in rows if row["step_id"] == target_step_id]
    if len(target_rows) != 1:
        raise ValueError("C500 prefill target step must occur exactly once")
    target_row = target_rows[0]
    if target_row["runtime_cudagraph_mode"] != "NONE":
        raise ValueError("C500 standalone prefill unexpectedly used Graph execution")
    if target_row["prefill_tokens"] < 1 or target_row["decode_tokens"] != 0:
        raise ValueError("C500 standalone prefill target has an invalid phase")

    replay_rows = [
        compact_mapping(value, "replay step comparison")
        for value in compact_sequence(replay["step_comparison"], "replay comparisons")
    ]
    replay_targets = [row for row in replay_rows if row["step_id"] == target_step_id]
    if len(replay_targets) != 1:
        raise ValueError("C500 prefill replay target step must occur exactly once")
    replay_target = replay_targets[0]
    fingerprint = compact_mapping(replay_target["step_fingerprint"], "step fingerprint")
    graph = compact_mapping(fingerprint["graph_execution"], "step graph execution")
    token_totals = compact_mapping(fingerprint["token_totals"], "step token totals")
    if graph["runtime_mode"] != "NONE":
        raise ValueError("C500 prefill fingerprint unexpectedly used Graph execution")
    if token_totals["prefill_tokens"] < 1 or token_totals["decode_tokens"] != 0:
        raise ValueError("C500 prefill fingerprint has an invalid phase")
    if token_totals["scheduled_tokens"] != target_row["scheduled_tokens"]:
        raise ValueError("C500 prefill scheduler/replay token totals differ")
    predicted_device_s = _finite(
        replay_target["predicted_device_s"], "prefill predicted device time"
    )
    expected_ranks = {
        _integer(value, "step graph rank")
        for value in compact_sequence(graph["rank_ids"], "step graph ranks")
    }

    context_paths = sorted(capture_dir.glob("graph_context.*.jsonl"))
    kernel_paths = sorted(capture_dir.glob("kernel_launch.*.jsonl"))
    if len(context_paths) != len(expected_ranks) or len(kernel_paths) != len(
        expected_ranks
    ):
        raise ValueError("C500 prefill correlation file count differs from expected ranks")
    kernel_by_pid = {
        _filename_pid(path, "kernel_launch"): path for path in kernel_paths
    }
    rank_inputs = {}
    for context_path in context_paths:
        pid, rank, contexts = _context_records(context_path, {target_step_id})
        if rank in rank_inputs:
            raise ValueError(f"duplicate C500 prefill worker rank {rank}")
        if pid not in kernel_by_pid:
            raise ValueError(f"missing C500 prefill kernel log for PID {pid}")
        launches, selected_launches = _kernel_launch_records(
            kernel_by_pid[pid], pid, target_step_id, contexts[target_step_id]
        )
        trace_path = capture_dir / "trace" / f"tracer_out-{pid}.json"
        if not trace_path.is_file():
            raise FileNotFoundError(
                f"missing C500 prefill mcTracer worker trace: {trace_path}"
            )
        rank_inputs[rank] = _raw_eager_rank_trace(
            trace_path, pid, rank, launches, selected_launches
        )
    if set(rank_inputs) != expected_ranks:
        raise ValueError(
            f"C500 prefill ranks differ: {sorted(rank_inputs)} != "
            f"{sorted(expected_ranks)}"
        )

    critical_path_s = max(
        rank_inputs[rank]["device_kernel_union_ns"] for rank in rank_inputs
    ) / 1_000_000_000
    return {
        "schema_version": 1,
        "kind": "c500_standalone_prefill_eager_mctracer",
        "comparison_status": (
            "candidate_device_critical_path_requires_target_pair_and_content_"
            "identity_confirmation"
        ),
        "content_identity": validation["content_identity"],
        "cell_id": validation["cell_id"],
        "step_id_mapping": dict(mapping),
        "graph_execution": dict(graph),
        "device_time_contract": (
            "max_per_rank_kernel_union_is_the_TP2_EP2_device_critical_path; "
            "host_service_and_cross_rank_clock_union_are_not_reported"
        ),
        "predicted_device_ms": predicted_device_s * 1_000,
        "target_kernel_critical_path_ms": critical_path_s * 1_000,
        "signed_device_critical_path_error_ms": (
            (predicted_device_s - critical_path_s) * 1_000
        ),
        "per_rank": [rank_inputs[rank] for rank in sorted(rank_inputs)],
        "source_sha256": {
            "validation": file_sha256(validation_path),
            "injection_replay": file_sha256(replay_path),
            "scheduler_trace": file_sha256(scheduler_path),
        },
    }
