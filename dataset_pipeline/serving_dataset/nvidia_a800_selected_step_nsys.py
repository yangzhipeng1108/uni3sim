"""Extract source-declared Graph or eager step intervals from A800 NSYS."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import statistics

from dataset_pipeline.serving_dataset.json_shape import (
    interval_union_ns,
    file_sha256,
)


_GLOBAL_PROCESS_MASK = -1 << 24




def _require_columns(
    connection: sqlite3.Connection, table: str, required: set[str]
) -> None:
    present = {
        row[1]
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"NSYS table {table} is missing columns: {missing}")


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None




def _correlated_device_intervals(
    connection: sqlite3.Connection,
    table: str,
    global_pid: int,
    correlation_ids: tuple[int, ...],
) -> list[tuple[int, int, int, int]]:
    """Return device activities launched by runtime calls inside one marker."""

    if not correlation_ids:
        raise ValueError("NSYS marker contains no runtime correlations")
    placeholders = ",".join("?" for _ in correlation_ids)
    rows = connection.execute(
        f"""
        SELECT start, end, deviceId, correlationId
        FROM {table}
        WHERE globalPid = ?
          AND correlationId IN ({placeholders})
        ORDER BY start
        """,
        (global_pid, *correlation_ids),
    ).fetchall()
    result = []
    for start, end, device_id, correlation_id in rows:
        if end <= start:
            raise ValueError(f"NSYS {table} device interval is invalid")
        result.append((start, end, device_id, correlation_id))
    return result


def _graph_node_kernel_groups(
    connection: sqlite3.Connection,
    global_pid: int,
    correlation_id: int,
) -> tuple[list[tuple[int, int]], int, int, list[dict[str, object]]]:
    """Return one node-traced Graph launch with exact kernel launch groups."""

    rows = connection.execute(
        """
        SELECT k.start, k.end, k.deviceId, k.graphNodeId, k.graphId,
               name.value, k.gridX, k.gridY, k.gridZ,
               k.blockX, k.blockY, k.blockZ,
               k.staticSharedMemory, k.dynamicSharedMemory
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds name ON k.shortName = name.id
        WHERE k.globalPid = ? AND k.correlationId = ?
        ORDER BY k.start
        """,
        (global_pid, correlation_id),
    ).fetchall()
    if not rows:
        raise ValueError("NSYS node-traced CUDA Graph contains no kernel nodes")
    devices = {row[2] for row in rows}
    graph_ids = {row[4] for row in rows}
    if None in devices or len(devices) != 1:
        raise ValueError("NSYS node-traced CUDA Graph device is ambiguous")
    if None in graph_ids or len(graph_ids) != 1:
        raise ValueError("NSYS node-traced CUDA Graph identity is ambiguous")
    groups: dict[str, dict[str, object]] = {}
    intervals = []
    for row in rows:
        (start, end, _device_id, graph_node_id, _graph_id, name,
         grid_x, grid_y, grid_z, block_x, block_y, block_z,
         static_shared, dynamic_shared) = row
        if end <= start:
            raise ValueError("NSYS node-traced CUDA Graph kernel interval is invalid")
        if graph_node_id is None:
            raise ValueError("NSYS node-traced CUDA Graph kernel has no node identity")
        if not isinstance(name, str) or not name:
            raise ValueError("NSYS node-traced CUDA Graph kernel name is invalid")
        intervals.append((start, end))
        launch = {
            "grid": [grid_x, grid_y, grid_z],
            "block": [block_x, block_y, block_z],
            "static_shared_bytes": static_shared,
            "dynamic_shared_bytes": dynamic_shared,
        }
        signature = {"name": name, "launch": launch}
        signature_key = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        if signature_key not in groups:
            groups[signature_key] = {
                **signature,
                "count": 0,
                "additive_duration_ns": 0,
            }
        groups[signature_key]["count"] += 1
        groups[signature_key]["additive_duration_ns"] += end - start
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: (
            -group["additive_duration_ns"],
            group["name"],
            json.dumps(group["launch"], sort_keys=True),
        ),
    )
    return intervals, devices.pop(), graph_ids.pop(), ordered_groups


def analyze_a800_selected_step_nsys(
    sqlite_path: Path,
    scheduler_trace_path: Path,
    step_ids: list[str],
    expected_rank_count: int,
    *,
    dispatch_mode: str = "graph",
) -> dict[str, object]:
    """Correlate each rank's NVTX step range to its device tasks."""

    path = sqlite_path.resolve(strict=True)
    trace_path = scheduler_trace_path.resolve(strict=True)
    if not step_ids or len(set(step_ids)) != len(step_ids):
        raise ValueError("step IDs must be non-empty and unique")
    if isinstance(expected_rank_count, bool) or expected_rank_count < 1:
        raise ValueError("expected rank count must be a positive integer")
    if dispatch_mode not in ("graph", "eager"):
        raise ValueError("A800 selected-step dispatch mode must be graph or eager")
    connection = sqlite3.connect(path)
    try:
        _require_columns(
            connection,
            "NVTX_EVENTS",
            {"start", "end", "eventType", "text", "textId", "globalTid"},
        )
        _require_columns(connection, "StringIds", {"id", "value"})
        _require_columns(connection, "PROCESSES", {"globalPid", "pid"})
        _require_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            {"start", "end", "globalTid", "correlationId", "nameId"},
        )
        has_graph_trace = _table_exists(
            connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
        )
        if has_graph_trace:
            _require_columns(
                connection,
                "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
                {
                    "start",
                    "end",
                    "deviceId",
                    "correlationId",
                    "globalPid",
                    "graphId",
                    "graphExecId",
                },
            )
            graph_trace_mode = "graph"
        else:
            _require_columns(
                connection,
                "CUPTI_ACTIVITY_KIND_KERNEL",
                {
                    "start", "end", "deviceId", "correlationId", "globalPid",
                    "graphNodeId", "graphId", "shortName",
                    "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ",
                    "staticSharedMemory", "dynamicSharedMemory",
                },
            )
            graph_trace_mode = "node"
        for table in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        ):
            _require_columns(
                connection,
                table,
                {"start", "end", "deviceId", "correlationId", "globalPid"},
            )
        trace_rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        trace_indexes = {row["step_id"]: index for index, row in enumerate(trace_rows)}
        if len(trace_indexes) != len(trace_rows):
            raise ValueError("Scheduler trace contains duplicate step IDs")
        steps = []
        for step_id in step_ids:
            if step_id not in trace_indexes:
                raise ValueError(f"Scheduler trace is missing step {step_id!r}")
            trace_index = trace_indexes[step_id]
            trace_row = trace_rows[trace_index]
            rank_observations = {
                observation["rank"]: observation
                for observation in trace_row["rank_observations"]
            }
            if len(rank_observations) != expected_rank_count:
                raise ValueError(
                    f"Scheduler trace rank count disagrees for {step_id!r}"
                )
            marker = f"midlayer-step:{step_id}"
            marker_rows = connection.execute(
                """
                SELECT n.start, n.end, n.globalTid, p.pid
                FROM NVTX_EVENTS n
                LEFT JOIN StringIds text_value ON n.textId = text_value.id
                JOIN PROCESSES p
                  ON p.globalPid = (n.globalTid & ?)
                WHERE (n.text = ? OR text_value.value = ?)
                  AND n.eventType IN (59, 70)
                  AND n.endGlobalTid IS NULL
                ORDER BY p.pid
                """,
                (_GLOBAL_PROCESS_MASK, marker, marker),
            ).fetchall()
            if len(marker_rows) != expected_rank_count:
                raise ValueError(
                    f"NSYS marker {marker!r} rank count disagrees: "
                    f"{len(marker_rows)} != {expected_rank_count}"
                )
            ranks = []
            graph_intervals = []
            device_intervals = []
            nvtx_intervals = []
            clock_offsets = []
            for marker_start, marker_end, global_tid, pid in marker_rows:
                if marker_end is None or marker_end <= marker_start:
                    raise ValueError(f"NSYS marker {marker!r} is incomplete")
                graph_launches = connection.execute(
                    """
                    SELECT r.start, r.end, r.correlationId
                    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                    JOIN StringIds name ON r.nameId = name.id
                    WHERE r.globalTid = ?
                      AND r.start >= ? AND r.end <= ?
                      AND name.value = 'cudaGraphLaunch_v10000'
                    ORDER BY r.start
                    """,
                    (global_tid, marker_start, marker_end),
                ).fetchall()
                if dispatch_mode == "graph":
                    if len(graph_launches) != 1:
                        raise ValueError(
                            f"NSYS marker {marker!r} must contain exactly one "
                            f"CUDA Graph launch for pid {pid}, got {len(graph_launches)}"
                        )
                    launch_start, launch_end, correlation_id = graph_launches[0]
                else:
                    if graph_launches:
                        raise ValueError(
                            f"NSYS eager marker {marker!r} contains a CUDA Graph launch"
                        )
                    launch_start = launch_end = correlation_id = None
                global_pid = global_tid & _GLOBAL_PROCESS_MASK
                runtime_correlations = tuple(
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT DISTINCT correlationId
                        FROM CUPTI_ACTIVITY_KIND_RUNTIME
                        WHERE globalTid = ?
                          AND start >= ? AND end <= ?
                          AND correlationId IS NOT NULL
                        ORDER BY correlationId
                        """,
                        (global_tid, marker_start, marker_end),
                    ).fetchall()
                )
                if (dispatch_mode == "graph"
                        and correlation_id not in runtime_correlations):
                    raise ValueError(
                        f"NSYS Graph launch correlation is outside {marker!r}"
                    )
                if dispatch_mode == "eager":
                    if not all(
                            observation["runtime_cudagraph_mode"] == "NONE"
                            for observation in rank_observations.values()):
                        raise ValueError(
                            f"NSYS eager marker {marker!r} disagrees with source dispatch"
                        )
                    rank_graph_intervals = []
                    graph_kernel_groups = []
                    graph_start = graph_end = graph_id = graph_exec_id = None
                    device_id = None
                elif graph_trace_mode == "graph":
                    graph_rows = connection.execute(
                        """
                        SELECT start, end, deviceId, graphId, graphExecId
                        FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
                        WHERE globalPid = ? AND correlationId = ?
                        """,
                        (global_pid, correlation_id),
                    ).fetchall()
                    if len(graph_rows) != 1:
                        raise ValueError(
                            f"NSYS CUDA Graph correlation is not unique for "
                            f"{marker!r}, pid {pid}: {len(graph_rows)}"
                        )
                    graph_start, graph_end, device_id, graph_id, graph_exec_id = (
                        graph_rows[0]
                    )
                    if graph_end <= graph_start:
                        raise ValueError("NSYS CUDA Graph device interval is invalid")
                    rank_graph_intervals = [(graph_start, graph_end)]
                    graph_kernel_groups = []
                else:
                    (
                        rank_graph_intervals,
                        device_id,
                        graph_id,
                        graph_kernel_groups,
                    ) = _graph_node_kernel_groups(
                        connection, global_pid, correlation_id
                    )
                    graph_start = min(start for start, _ in rank_graph_intervals)
                    graph_end = max(end for _, end in rank_graph_intervals)
                    graph_exec_id = None
                graph_intervals.extend(rank_graph_intervals)
                rank_device_intervals = (
                    list(rank_graph_intervals)
                    if graph_trace_mode == "graph"
                    else []
                )
                task_counts = {
                    "graph": 1 if dispatch_mode == "graph" else 0,
                    "kernel": 0, "memcpy": 0, "memset": 0,
                }
                if graph_trace_mode == "node" and dispatch_mode == "graph":
                    task_counts["graph_kernel_node"] = len(rank_graph_intervals)
                activity_device_ids = set()
                for kind, table in (
                    ("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
                    ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
                    ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
                ):
                    activities = _correlated_device_intervals(
                        connection, table, global_pid, runtime_correlations
                    )
                    for start, end, activity_device_id, _ in activities:
                        if (device_id is not None
                                and activity_device_id != device_id):
                            raise ValueError(
                                f"NSYS {kind} device disagrees with Graph device"
                            )
                        activity_device_ids.add(activity_device_id)
                        rank_device_intervals.append((start, end))
                    task_counts[kind] = len(activities)
                if dispatch_mode == "eager":
                    if len(activity_device_ids) != 1 or not rank_device_intervals:
                        raise ValueError(
                            f"NSYS eager marker {marker!r} must bind one device"
                        )
                    device_id = next(iter(activity_device_ids))
                    graph_start = min(start for start, _ in rank_device_intervals)
                    graph_end = max(end for _, end in rank_device_intervals)
                if device_id not in rank_observations:
                    raise ValueError(
                        f"NSYS device {device_id} has no Scheduler rank observation"
                    )
                observation = rank_observations[device_id]
                clock_offsets.extend((
                    marker_start - observation["worker_execution_start_ns"],
                    marker_end - observation["worker_execution_end_ns"],
                ))
                device_intervals.extend(rank_device_intervals)
                nvtx_intervals.append((marker_start, marker_end))
                ranks.append({
                    "pid": pid,
                    "device_id": device_id,
                    "nvtx_worker_envelope_ms": (
                        marker_end - marker_start
                    ) / 1_000_000,
                    "cuda_graph_launch_api_ms": (
                        (launch_end - launch_start) / 1_000_000
                        if dispatch_mode == "graph" else None
                    ),
                    "device_graph_task_ms": (
                        interval_union_ns(rank_graph_intervals)
                    ) / 1_000_000,
                    "device_graph_span_ms": (
                        (graph_end - graph_start) / 1_000_000
                        if dispatch_mode == "graph" else None
                    ),
                    "device_task_union_ms": (
                        interval_union_ns(rank_device_intervals) / 1_000_000
                    ),
                    "device_task_counts": task_counts,
                    "graph_kernel_groups": graph_kernel_groups,
                    "correlation_id": correlation_id,
                    "graph_id": graph_id,
                    "graph_exec_id": graph_exec_id,
                    "device_start_ns": graph_start,
                    "device_end_ns": graph_end,
                })
            clock_offset_ns = round(statistics.median(clock_offsets))
            mapped_device_intervals = [
                (start - clock_offset_ns, end - clock_offset_ns)
                for start, end in device_intervals
            ]
            previous_update_start = (
                trace_row["schedule_end_ns"]
                if trace_index == 0
                else trace_rows[trace_index - 1]["update_start_ns"]
            )
            exposed_start_ns = max(
                trace_row["schedule_end_ns"], previous_update_start
            )
            exposed_end_ns = trace_row["update_start_ns"]
            if exposed_end_ns < exposed_start_ns:
                raise ValueError("Scheduler serialized service interval is negative")
            exposed_device_intervals = [
                (max(start, exposed_start_ns), min(end, exposed_end_ns))
                for start, end in mapped_device_intervals
                if max(start, exposed_start_ns) < min(end, exposed_end_ns)
            ]
            serialized_service_ns = exposed_end_ns - exposed_start_ns
            exposed_device_union_ns = interval_union_ns(
                exposed_device_intervals
            )
            steps.append({
                "step_id": step_id,
                "marker": marker,
                "rank_count": len(ranks),
                "ranks": ranks,
                "cross_rank_nvtx_union_ms": (
                    interval_union_ns(nvtx_intervals) / 1_000_000
                ),
                "cross_device_graph_task_union_ms": (
                    interval_union_ns(graph_intervals) / 1_000_000
                ),
                "cross_device_task_union_ms": (
                    interval_union_ns(device_intervals) / 1_000_000
                ),
                "clock_mapping": {
                    "method": "median_nvtx_to_worker_envelope_offset",
                    "nsys_minus_scheduler_ns": clock_offset_ns,
                    "max_anchor_residual_ns": max(
                        abs(offset - clock_offset_ns) for offset in clock_offsets
                    ),
                },
                "serialized_service_ms": serialized_service_ns / 1_000_000,
                "exposed_device_task_union_ms": (
                    exposed_device_union_ns / 1_000_000
                ),
                "uncovered_serialized_service_ms": (
                    serialized_service_ns - exposed_device_union_ns
                ) / 1_000_000,
                "runtime_inter_task_gap_ms": None,
            })
    finally:
        connection.close()
    return {
        "schema_version": 3,
        "kind": "a800_selected_step_nsys_timeline",
        "dispatch_mode": dispatch_mode,
        "cuda_graph_trace_mode": graph_trace_mode,
        "clock_domain": "nsys_global_timestamp_ns",
        "source_sqlite": str(path),
        "source_sqlite_sha256": file_sha256(path),
        "scheduler_trace": str(trace_path),
        "scheduler_trace_sha256": file_sha256(trace_path),
        "steps": steps,
    }
