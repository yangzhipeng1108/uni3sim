#!/usr/bin/env python3
"""Validate that a captured Serving trace is paced, real, and self-consistent."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    if not value:
        raise ValueError(f"{context} must not be empty")
    return value


def _require_number(value: Any, context: str, *, non_negative: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if non_negative and result < 0:
        raise ValueError(f"{context} must be non-negative")
    return result


def _require_integer(value: Any, context: str, *, non_negative: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if non_negative and value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _load_json(path: Path, context: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing {context}: {path}")
    with path.open() as stream:
        return json.load(stream)


def load_scheduler_records(path: Path) -> list[dict[str, Any]]:
    """Load a non-empty scheduler sidecar and validate its ordering contract."""

    if not path.is_file():
        raise FileNotFoundError(f"missing scheduler sidecar: {path}")
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"scheduler sidecar contains a blank line at {line_number}")
            record = _require_mapping(
                json.loads(line), f"scheduler record {line_number}")
            iteration = _require_integer(
                record["iter_id"], f"scheduler record {line_number}.iter_id",
                non_negative=True)
            timestamp = _require_number(
                record["ts_us"], f"scheduler record {line_number}.ts_us",
                non_negative=True)
            total_tokens = _require_integer(
                record["total_tokens"],
                f"scheduler record {line_number}.total_tokens",
                non_negative=True)
            scheduled = _require_mapping(
                record["scheduled"],
                f"scheduler record {line_number}.scheduled")
            records.append({
                "iter_id": iteration,
                "ts_us": timestamp,
                "total_tokens": total_tokens,
                "scheduled": scheduled,
            })
    if not records:
        raise ValueError(f"scheduler sidecar is empty: {path}")
    iterations = [record["iter_id"] for record in records]
    if len(iterations) != len(set(iterations)):
        raise ValueError(f"scheduler sidecar contains duplicate iter_id: {path}")
    if any(later <= earlier for earlier, later in zip(
            iterations, iterations[1:])):
        raise ValueError(
            f"scheduler iter_id values must strictly increase: {path}")
    timestamps = [record["ts_us"] for record in records]
    if any(later <= earlier for earlier, later in zip(
            timestamps, timestamps[1:])):
        raise ValueError(
            f"scheduler timestamps must strictly increase: {path}")
    return records


def validate_case(case_dir: Path) -> list[str]:
    """Return semantic validation failures; malformed inputs raise immediately."""

    arrival = _require_mapping(
        _load_json(case_dir / "arrival_schedule.json", "arrival schedule"),
        "arrival schedule")
    arrival_times = [
        _require_number(value, f"arrival_times_s[{index}]", non_negative=True)
        for index, value in enumerate(_require_list(
            arrival["arrival_times_s"], "arrival_times_s"))
    ]
    if any(later < earlier for earlier, later in zip(
            arrival_times, arrival_times[1:])):
        raise ValueError("arrival_times_s must be non-decreasing")
    arrival_span = arrival_times[-1] - arrival_times[0]
    if arrival_span <= 0:
        raise ValueError("arrival schedule must have a positive time span")
    qps = _require_number(arrival["qps"], "arrival qps", non_negative=True)
    print(
        f"[arrival] qps={qps} span={arrival_span:.1f}s "
        f"n={len(arrival_times)}")

    sidecar = load_scheduler_records(
        case_dir / "plugin_trace.json_scheduler.jsonl")
    active = [record for record in sidecar if record["total_tokens"] > 0]
    if not active:
        raise ValueError("scheduler sidecar contains no active steps")
    first_scheduled_at: dict[str, float] = {}
    for record in active:
        if not record["scheduled"]:
            raise ValueError(
                f"active scheduler iteration {record['iter_id']} has no requests")
        for request_id in record["scheduled"]:
            if not isinstance(request_id, str) or not request_id:
                raise TypeError("scheduled request IDs must be non-empty strings")
            if request_id not in first_scheduled_at:
                first_scheduled_at[request_id] = record["ts_us"]
    admission_times = sorted(first_scheduled_at.values())
    admission_span = (admission_times[-1] - admission_times[0]) / 1e6
    first_ids = list(active[0]["scheduled"])
    print(
        f"[sidecar] admission_span={admission_span:.1f}s "
        f"n_active_steps={len(active)} iter0={first_ids}")

    problems = []
    pacing_ratio = admission_span / arrival_span
    pacing_ok = 0.8 <= pacing_ratio <= 1.2
    print(
        f"[check 1] PACED: admission/arrival ratio={pacing_ratio:.2f} "
        f"({'PASS' if pacing_ok else 'FAIL'})")
    if not pacing_ok:
        problems.append("scheduler admission times do not match the arrival pace")

    if all(re.fullmatch(r"\d+", request_id) for request_id in first_ids):
        problems.append("numeric request IDs identify a batch capture")
        print("[check 1b] ID FORMAT: FAIL")
    else:
        print("[check 1b] ID FORMAT: PASS")

    warmup_ok = any(request_id.startswith("warmup") for request_id in first_ids)
    print(f"[check 2] WARMUP: {'PASS' if warmup_ok else 'FAIL'}")
    if not warmup_ok:
        problems.append("warmup request is not retained in iteration zero")

    metrics = _require_mapping(
        _load_json(case_dir / "serving_metrics.json", "Serving metrics"),
        "Serving metrics")
    per_request = [
        _require_mapping(value, f"per_request[{index}]")
        for index, value in enumerate(_require_list(
            metrics["per_request"], "per_request"))
    ]
    output_lengths = [
        _require_integer(
            record["n_output_tokens"],
            f"per_request[{index}].n_output_tokens",
            non_negative=True)
        for index, record in enumerate(per_request)
    ]
    send_times = [
        _require_number(
            record["send_time_s"], f"per_request[{index}].send_time_s",
            non_negative=True)
        for index, record in enumerate(per_request)
    ]
    if any(later < earlier for earlier, later in zip(send_times, send_times[1:])):
        raise ValueError("per-request send_time_s values must be non-decreasing")
    send_span = send_times[-1] - send_times[0]
    diverse_output = len(set(output_lengths)) > 1
    print(
        f"[GT] n={len(per_request)} output_len_unique="
        f"{len(set(output_lengths))} send_span={send_span:.1f}s")
    print(f"[check 3] REAL WORKLOAD: {'PASS' if diverse_output else 'FAIL'}")
    if not diverse_output:
        problems.append("all requests have the same output length")

    same_run_ratio = send_span / arrival_span
    same_run_ok = 0.8 <= same_run_ratio <= 1.2
    print(
        f"[check 4] SAME RUN: send/arrival ratio={same_run_ratio:.2f} "
        f"({'PASS' if same_run_ok else 'FAIL'})")
    if not same_run_ok:
        problems.append("Serving metrics and arrival schedule are not from one run")
    return problems


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    problems = validate_case(_parse_args(argv).case_dir)
    if problems:
        print(f"\nRESULT: FAIL - {problems}")
        return 1
    print("\nRESULT: PASS")
    return 0
