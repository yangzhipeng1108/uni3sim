"""I/O and step-boundary helpers for retained serving-trace alignment.

These functions consume captured nsys/macaprof artifacts only.  They live in
the validation package so production serving and kernel prediction never
depend on runtime kernel names or archived trace-synthesis code.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
from pathlib import Path


def _require_mapping(value, context):
    if not isinstance(value, dict):
        raise TypeError(
            f"{context} must be a mapping, got {type(value).__name__}")
    return value


def _require_integer(value, context, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{context} must be an integer, got {type(value).__name__}")
    if positive and value <= 0:
        raise ValueError(f"{context} must be positive, got {value}")
    if not positive and value < 0:
        raise ValueError(f"{context} must be non-negative, got {value}")
    return value


def _device_id(value):
    return _require_integer(value, "device_id", positive=False)


def _validated_intervals(values, context):
    if not isinstance(values, list):
        raise TypeError(f"{context} must be a list")
    result = []
    for index, item in enumerate(values):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError(
                f"{context}[{index}] must contain start and end")
        start = _require_integer(item[0], f"{context}[{index}].start")
        end = _require_integer(item[1], f"{context}[{index}].end")
        if end < start:
            raise ValueError(
                f"{context}[{index}] end precedes start: {start} -> {end}")
        result.append((start, end))
    return result


def load_scheduler_sidecar(case_dir):
    """Return sorted scheduler iteration IDs and their wall-clock timestamps."""
    path = Path(case_dir) / "plugin_trace.json_scheduler.jsonl"
    if not path.exists():
        return None, None
    entries = []
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = _require_mapping(
                json.loads(line), f"scheduler record in {path}")
            iteration = _require_integer(
                record["iter_id"], "scheduler iter_id", positive=False)
            timestamp_us = _require_integer(
                record["ts_us"], "scheduler ts_us", positive=True)
            entries.append((iteration, timestamp_us))
    if not entries:
        raise ValueError(f"scheduler sidecar contains no records: {path}")
    entries.sort()
    iterations = [entry[0] for entry in entries]
    timestamps = [entry[1] for entry in entries]
    if len(iterations) != len(set(iterations)):
        raise ValueError(f"scheduler sidecar contains duplicate iter_id: {path}")
    if any(later <= earlier for earlier, later in zip(
            timestamps, timestamps[1:])):
        raise ValueError(
            f"scheduler sidecar timestamps must strictly increase: {path}")
    return iterations, timestamps


def idle_iteration_ids(case_dir):
    """Return scheduler iterations which executed no model-forward tokens."""
    path = Path(case_dir) / "plugin_trace.json_scheduler.jsonl"
    if not path.exists():
        return set()
    idle = set()
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = _require_mapping(
                json.loads(line), f"scheduler record in {path}")
            iteration = _require_integer(
                record["iter_id"], "scheduler iter_id", positive=False)
            _require_integer(
                record["ts_us"], "scheduler ts_us", positive=True)
            total_tokens = _require_integer(
                record["total_tokens"], "scheduler total_tokens",
                positive=False)
            if total_tokens == 0:
                idle.add(iteration)
    return idle


def load_nsys_json(case_dir, device_id=0):
    """Load a pre-extracted ``nsys_kernels_full.json`` kernel sequence."""
    path = Path(case_dir) / "nsys_kernels_full.json"
    if not path.exists():
        return None
    device = _device_id(device_id)
    with path.open() as stream:
        data = _require_mapping(json.load(stream), f"nsys trace {path}")
    has_per_rank = "per_rank" in data
    has_kernels = "kernels" in data
    if has_per_rank == has_kernels:
        raise ValueError(
            f"nsys trace must contain exactly one of per_rank or kernels: {path}")
    if has_per_rank:
        per_rank = _require_mapping(data["per_rank"], f"nsys per_rank {path}")
        rank_keys = [key for key in per_rank if key == str(device)]
        if len(rank_keys) != 1:
            raise KeyError(
                f"nsys trace has no unique rank for device {device}: {path}")
        rank = per_rank[rank_keys[0]]
        if isinstance(rank, dict):
            kernels = rank["kernels"]
        elif isinstance(rank, list):
            kernels = rank
        else:
            raise TypeError(
                f"nsys rank {device} must be a mapping or list")
    elif has_kernels:
        kernels = data["kernels"]
    else:
        raise AssertionError("unreachable nsys trace schema branch")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError(f"nsys kernel list is empty: {path}")
    result = []
    for index, raw_entry in enumerate(kernels):
        entry = _require_mapping(
            raw_entry, f"nsys kernel[{index}] in {path}")
        grid = entry["grid"]
        block = entry["block"]
        if (not isinstance(grid, (list, tuple))
                or not isinstance(block, (list, tuple))
                or len(grid) != 3 or len(block) != 3):
            raise ValueError(f"nsys launch geometry is incomplete: {entry!r}")
        grid3 = tuple(
            _require_integer(value, f"nsys kernel[{index}] grid", positive=True)
            for value in grid)
        block3 = tuple(
            _require_integer(value, f"nsys kernel[{index}] block", positive=True)
            for value in block)
        timestamp_ns = _require_integer(
            entry["ts_ns"], f"nsys kernel[{index}] ts_ns", positive=False)
        shared_memory = _require_integer(
            entry["smem"], f"nsys kernel[{index}] smem", positive=False)
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise TypeError(f"nsys kernel[{index}] name must be non-empty")
        result.append((
            timestamp_ns,
            name,
            grid3,
            block3,
            shared_memory,
        ))
    result.sort(key=lambda item: item[0])
    return result


def nsys_time_bounds(case_dir, device_id=None):
    """Return the first model-kernel start and final captured kernel end."""
    case_dir = Path(case_dir)
    sqlite_path = case_dir / "nsys_profile.sqlite"
    device = _device_id(device_id) if device_id is not None else 0
    if not sqlite_path.exists():
        kernels = load_nsys_json(case_dir, device)
        if not kernels:
            return None, None
        start = next(
            (row[0] for row in kernels if "cross_device_reduce" in row[1]),
            None,
        )
        if start is None:
            start = next(
                (row[0] for row in kernels
                 if "rms_norm_kernel" in row[1]
                 or "fused_add_rms_norm" in row[1]),
                None,
            )
        if start is None:
            raise ValueError(
                "cannot identify the first model kernel in "
                f"{case_dir / 'nsys_kernels_full.json'}")
        return start, max(row[0] for row in kernels)

    connection = sqlite3.connect(str(sqlite_path))
    try:
        row = connection.execute(
            "SELECT MIN(k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.demangledName=s.id "
            "WHERE k.deviceId=? AND s.value LIKE '%cross_device_reduce%'",
            (device,),
        ).fetchone()
        if row is None or row[0] is None:
            row = connection.execute(
                "SELECT MIN(k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.demangledName=s.id "
                "WHERE k.deviceId=? AND (s.value LIKE '%rms_norm_kernel%' "
                "OR s.value LIKE '%fused_add_rms_norm%')",
                (device,),
            ).fetchone()
        if row is None or row[0] is None:
            raise ValueError(
                "cannot identify the first model kernel in "
                f"{sqlite_path} for device {device}")
        last = connection.execute(
            "SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=?",
            (device,),
        ).fetchone()
        if last is None or last[0] is None:
            raise ValueError(
                f"no kernel end timestamp in {sqlite_path} for device {device}")
        return row[0], last[0]
    finally:
        connection.close()


def timestamp_boundaries(sidecar_timestamps_us, first_kernel_ns, last_kernel_ns):
    """Map sidecar timestamps to the profiler clock using relative offsets."""
    first_timestamp = sidecar_timestamps_us[0]
    boundaries = [
        first_kernel_ns + (timestamp - first_timestamp) * 1000.0
        for timestamp in sidecar_timestamps_us
    ]
    boundaries.append(max(last_kernel_ns + 1.0, boundaries[-1] + 1.0))
    return boundaries


def _mctracer_synchronizations(case_dir, device_id=0):
    """Extract C500 step synchronization intervals from macaprof JSON."""
    case_dir = Path(case_dir)
    rank = _device_id(device_id) if device_id is not None else 0
    cache = case_dir / f".mctracer_sync_ends_{rank}.json"
    if cache.exists():
        with cache.open() as stream:
            cached = json.load(stream)
        if cached:
            return _validated_intervals(cached, f"mctracer cache {cache}")

    candidates = sorted(
        glob.glob(str(case_dir / "mctracer_out" / "mctracer-*.json")),
        key=os.path.getsize,
        reverse=True,
    )
    large = [path for path in candidates if os.path.getsize(path) > 50_000_000]
    if not large:
        large = candidates
    if not large:
        return None
    selected = large[0]
    if len(large) > 1 and device_id is not None:
        device_pattern = re.compile(rb'"device_id":\s*(\d+)')
        best_score = -1
        for candidate in large:
            score = 0
            sampled = 0
            with open(candidate, "rb", buffering=1 << 20) as stream:
                for raw_line in stream:
                    for match in device_pattern.finditer(raw_line):
                        sampled += 1
                        score += int(match.group(1)) == rank
                    if sampled > 20_000:
                        break
            if score > best_score:
                selected, best_score = candidate, score

    timestamp_pattern = re.compile(r'"ts":\s*(\d+)')
    duration_pattern = re.compile(r'"dur":\s*(\d+)')
    name_pattern = re.compile(r'"name":\s*"([^"]+)"')
    wanted = {"mcEventSynchronize", "mcDeviceSynchronize", "mcStreamSynchronize"}
    events = []
    last_timestamp = last_duration = None
    with open(selected, "rb", buffering=1 << 20) as stream:
        for raw_line in stream:
            line = raw_line.decode("utf-8", "replace")
            if line.strip() in ("{", ",{"):
                last_timestamp = last_duration = None
            name_match = name_pattern.search(line)
            if (name_match and name_match.group(1) in wanted
                    and last_timestamp is not None):
                if last_duration is None:
                    raise ValueError(
                        f"synchronization event has no duration in {selected}")
                events.append((name_match.group(1), last_timestamp, last_duration))
                continue
            timestamp_match = timestamp_pattern.search(line)
            duration_match = duration_pattern.search(line)
            if timestamp_match:
                last_timestamp = int(timestamp_match.group(1))
            if duration_match:
                last_duration = int(duration_match.group(1))

    intervals = [
        (timestamp, timestamp + duration)
        for name, timestamp, duration in events
        if name in ("mcEventSynchronize", "mcDeviceSynchronize")
    ]
    if not intervals:
        intervals = [
            (timestamp, timestamp + duration)
            for name, timestamp, duration in events
            if name == "mcStreamSynchronize"
        ]
    intervals.sort()
    with cache.open("w") as stream:
        json.dump(intervals, stream)
    if intervals:
        return intervals
    return None


def _sqlite_synchronizations(case_dir, device_id=None):
    sqlite_path = Path(case_dir) / "nsys_profile.sqlite"
    if not sqlite_path.exists():
        return None
    connection = sqlite3.connect(str(sqlite_path))
    try:
        if device_id is None:
            rows = connection.execute(
                "SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON r.nameId=s.id "
                "WHERE s.value LIKE 'cudaStreamSynchronize%' ORDER BY r.start"
            ).fetchall()
            return _validated_intervals(
                [list(row) for row in rows],
                f"nsys synchronizations in {sqlite_path}")
        ownership = {}
        for global_tid, device, count in connection.execute(
                "SELECT r.globalTid, k.deviceId, COUNT(*) "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON r.nameId=s.id "
                "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId "
                "WHERE s.value LIKE 'cudaLaunchKernel%' "
                "GROUP BY r.globalTid, k.deviceId").fetchall():
            if global_tid not in ownership or ownership[global_tid][1] < count:
                ownership[global_tid] = (device, count)
        threads = [
            global_tid for global_tid, (device, _) in ownership.items()
            if device == _device_id(device_id)
        ]
        if not threads:
            return None
        placeholders = ",".join("?" for _ in threads)
        rows = connection.execute(
            "SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId=s.id "
            f"WHERE s.value LIKE 'cudaStreamSynchronize%' "
            f"AND r.globalTid IN ({placeholders}) ORDER BY r.start",
            threads,
        ).fetchall()
        return _validated_intervals(
            [list(row) for row in rows],
            f"nsys synchronizations in {sqlite_path}")
    except sqlite3.OperationalError as error:
        raise RuntimeError(
            f"failed to read synchronization data from {sqlite_path}") from error
    finally:
        connection.close()


def synchronization_boundaries(case_dir, iterations, device_id=None):
    """Return structural profiler boundaries and their aligned iterations."""
    synchronizations = _sqlite_synchronizations(case_dir, device_id)
    from_mctracer = False
    if not synchronizations:
        synchronizations = _mctracer_synchronizations(case_dir, device_id)
        from_mctracer = bool(synchronizations)
        if synchronizations:
            first, last = nsys_time_bounds(case_dir, device_id)
            if first is not None:
                synchronizations = [item for item in synchronizations if item[1] >= first]
            if last is not None:
                synchronizations = [item for item in synchronizations if item[0] <= last]
    if not synchronizations:
        return None
    if not iterations:
        raise ValueError(
            f"synchronization markers exist but scheduler iterations are empty: "
            f"{case_dir}")

    idle = idle_iteration_ids(case_dir)
    iterations = [iteration for iteration in iterations if iteration not in idle]
    if not iterations:
        return None

    if len(synchronizations) > 3 * len(iterations):
        candidates = [
            index for index in range(len(synchronizations) - 1)
            if synchronizations[index + 1][0] - synchronizations[index][1] >= 3_000_000
        ]
        kept = []
        for index in candidates:
            if kept and index - kept[-1] < 180:
                continue
            kept.append(index)
        ends = [synchronizations[index][1] for index in kept]
        if not 0.9 * len(iterations) <= len(ends) <= 1.1 * len(iterations):
            raise ValueError(
                "filtered synchronization count does not match scheduler "
                f"iterations for {case_dir}: {len(ends)} vs {len(iterations)}")
        ep_filtered = True
    else:
        ends = [item[1] for item in synchronizations]
        ep_filtered = False

    if not ep_filtered:
        count_matches = len(ends) in (len(iterations), len(iterations) - 1)
        prefix_matches = from_mctracer and 2 <= len(ends) <= len(iterations)
        if not (count_matches or prefix_matches):
            raise ValueError(
                "synchronization count does not match scheduler iterations "
                f"for {case_dir}: {len(ends)} vs {len(iterations)}")
    aligned_iterations = list(iterations[:len(ends)])
    first, _ = nsys_time_bounds(case_dir, device_id)
    if first is None:
        raise ValueError(
            f"synchronization markers exist but profiler has no kernel bounds: "
            f"{case_dir}")
    return [first] + ends, aligned_iterations
