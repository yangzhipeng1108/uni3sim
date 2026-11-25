#!/usr/bin/env python3
"""Measure eager NPU submission and ACL Graph replay without a profiler."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import traceback

import torch
import torch_npu


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
CHAIN_LENGTHS = (1, 8, 64, 512, 4096)
REQUIRED_CHAIN_LENGTHS = frozenset((1, 8, 64, 512))
MODES = ("eager", "acl_graph_replay")
TRIALS = 30
WARMUP_REPEATS = 5
TENSOR_NUMEL = 4096
SCHEMA_VERSION = 2


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _append_json_line(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summary(samples):
    fields = (
        "host_submit_ns",
        "host_submit_thread_cpu_ns",
        "host_submit_non_cpu_wall_ns",
        "event_record_host_ns",
        "end_synchronize_ns",
        "host_end_to_end_ns",
        "device_elapsed_ns",
        "host_minus_device_ns",
        "host_submit_ns_per_logical_op",
        "device_elapsed_ns_per_logical_op",
    )
    result = {}
    for field in fields:
        values = [float(sample[field]) for sample in samples]
        result[field] = {
            "min": min(values),
            "p10": _percentile(values, 0.10),
            "median": statistics.median(values),
            "p90": _percentile(values, 0.90),
            "max": max(values),
        }
    return result


def _cann_identity():
    ascend_home = Path(os.environ["ASCEND_HOME_PATH"])
    version_path = ascend_home / "opp" / "version.info"
    return {
        "ascend_home_path": str(ascend_home),
        "opp_version_path": str(version_path),
        "opp_version_text": version_path.read_text(encoding="utf-8"),
        "opp_version_sha256": _sha256(version_path),
    }


def _device_identity(device_id):
    properties = torch.npu.get_device_properties(device_id)
    return {
        "device_id": device_id,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "cube_core_count": int(properties.cube_core_num),
        "vector_core_count": int(properties.vector_core_num),
        "properties_repr": repr(properties),
    }


def _ordered_cells(trial_index, available_lengths):
    base = tuple(
        (mode, chain_length)
        for chain_length in available_lengths
        for mode in MODES
    )
    rotation = trial_index % len(base)
    rotated = base[rotation:] + base[:rotation]
    return rotated if trial_index % 2 == 0 else tuple(reversed(rotated))


def _validate_output_dir(path):
    resolved = path.resolve()
    owner_size = len(OUTPUT_OWNER_PARTS)
    owner_offsets = tuple(
        index
        for index in range(len(resolved.parts) - owner_size + 1)
        if resolved.parts[index:index + owner_size] == OUTPUT_OWNER_PARTS
    )
    if len(owner_offsets) != 1:
        raise ValueError(
            "--output-dir must be under exactly one "
            "profiler_output/microbench/ascend_910b owner")
    if len(resolved.parts) <= owner_offsets[0] + owner_size:
        raise ValueError("--output-dir must include a run-specific directory")
    if resolved.exists():
        raise FileExistsError("--output-dir must not already exist")
    return resolved


def _run_chain(left, right, increment, chain_length):
    source = left
    destination = right
    for _ in range(chain_length):
        torch.add(source, increment, out=destination)
        source, destination = destination, source
    return source


def _capture_graph(stream, pool, left, right, increment, chain_length):
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=pool, stream=stream):
        _run_chain(left, right, increment, chain_length)
    return graph


def _measure(stream, mode, chain_length, left, right, increment, graph):
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record(stream)
    submit_started_ns = time.perf_counter_ns()
    submit_started_thread_cpu_ns = time.thread_time_ns()
    if mode == "eager":
        _run_chain(left, right, increment, chain_length)
    elif mode == "acl_graph_replay":
        if graph is None:
            raise RuntimeError("ACL Graph was not captured for this cell")
        graph.replay()
    else:
        raise ValueError(f"unsupported mode {mode}")
    submit_ended_thread_cpu_ns = time.thread_time_ns()
    submit_ended_ns = time.perf_counter_ns()
    end_event.record(stream)
    synchronize_started_ns = time.perf_counter_ns()
    end_event.synchronize()
    synchronize_ended_ns = time.perf_counter_ns()
    device_elapsed_ns = float(start_event.elapsed_time(end_event)) * 1_000_000.0
    host_submit_ns = submit_ended_ns - submit_started_ns
    host_submit_thread_cpu_ns = (
        submit_ended_thread_cpu_ns - submit_started_thread_cpu_ns)
    event_record_host_ns = synchronize_started_ns - submit_ended_ns
    end_synchronize_ns = synchronize_ended_ns - synchronize_started_ns
    host_end_to_end_ns = synchronize_ended_ns - submit_started_ns
    values = (
        host_submit_ns,
        host_submit_thread_cpu_ns,
        event_record_host_ns,
        end_synchronize_ns,
        host_end_to_end_ns,
        device_elapsed_ns,
    )
    if any(not math.isfinite(float(value)) or value <= 0 for value in values):
        raise RuntimeError("timing sample contains a non-positive value")
    return {
        "host_submit_ns": host_submit_ns,
        "host_submit_thread_cpu_ns": host_submit_thread_cpu_ns,
        "host_submit_non_cpu_wall_ns": (
            host_submit_ns - host_submit_thread_cpu_ns),
        "event_record_host_ns": event_record_host_ns,
        "end_synchronize_ns": end_synchronize_ns,
        "host_end_to_end_ns": host_end_to_end_ns,
        "device_elapsed_ns": device_elapsed_ns,
        "host_minus_device_ns": host_end_to_end_ns - device_elapsed_ns,
        "host_submit_ns_per_logical_op": host_submit_ns / chain_length,
        "device_elapsed_ns_per_logical_op": device_elapsed_ns / chain_length,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--container-name", required=True)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if BASELINE_PATTERN.fullmatch(args.baseline_commit) is None:
        parser.error("--baseline-commit must be a full lowercase Git commit")
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    if len(args.expected_source_sha256) != 64:
        parser.error("--expected-source-sha256 must be a SHA256 digest")
    args.output_dir = _validate_output_dir(args.output_dir)
    return args


def main():
    args = _parse_args()
    source_path = Path(__file__).resolve()
    actual_source_sha256 = _sha256(source_path)
    if actual_source_sha256 != args.expected_source_sha256:
        raise RuntimeError(
            "runner source SHA256 differs from the audited snapshot")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "manifest.json"
    trials_path = args.output_dir / "trials.jsonl"
    summary_path = args.output_dir / "summary.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "initializing",
        "baseline_commit": args.baseline_commit,
        "source": {
            "path": str(source_path),
            "sha256": actual_source_sha256,
        },
        "command": [sys.executable, *sys.argv],
        "pid": os.getpid(),
        "container_name": args.container_name,
        "profiler": None,
        "device": None,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "cann": None,
        },
        "contract": {
            "stream_count": 1,
            "preallocated_tensors": True,
            "operation": "torch.add(source, increment, out=destination)",
            "tensor_numel": TENSOR_NUMEL,
            "dtype": "float32",
            "chain_lengths": list(CHAIN_LENGTHS),
            "required_chain_lengths": sorted(REQUIRED_CHAIN_LENGTHS),
            "trials_per_cell": TRIALS,
            "warmup_repeats": WARMUP_REPEATS,
            "schedule": "paired_forward_reverse_rotations",
            "event_clock": "torch.npu.Event(enable_timing=True)",
            "host_clock": "time.perf_counter_ns",
            "host_thread_cpu_clock": "time.thread_time_ns",
            "graph_api": "torch.npu.graph(NPUGraph, pool, explicit_stream)",
            "graph_replay_pre_synchronize": False,
        },
        "graph_capture_failures": [],
        "available_chain_lengths": None,
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "failure": None,
        "outputs": {
            "trials": str(trials_path),
            "summary": str(summary_path),
        },
    }
    _write_json(manifest_path, manifest)
    try:
        manifest["environment"]["cann"] = _cann_identity()
        torch.npu.set_device(args.device_id)
        manifest["device"] = _device_identity(args.device_id)
        stream = torch.npu.Stream(device=args.device_id)
        pool = torch.npu.graph_pool_handle()
        with torch.npu.stream(stream):
            left = torch.zeros(
                TENSOR_NUMEL, dtype=torch.float32,
                device=f"npu:{args.device_id}")
            right = torch.empty_like(left)
            increment = torch.ones_like(left)
            for chain_length in CHAIN_LENGTHS:
                for _ in range(WARMUP_REPEATS):
                    _run_chain(left, right, increment, chain_length)
                stream.synchronize()

            graphs = {}
            for chain_length in CHAIN_LENGTHS:
                try:
                    graph = _capture_graph(
                        stream, pool, left, right, increment, chain_length)
                    graph.replay()
                    stream.synchronize()
                    graphs[chain_length] = graph
                except Exception as error:
                    failure = {
                        "chain_length": chain_length,
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    manifest["graph_capture_failures"].append(failure)
                    _write_json(manifest_path, manifest)
                    if chain_length in REQUIRED_CHAIN_LENGTHS:
                        raise RuntimeError(
                            f"required ACL Graph capture failed for {chain_length}"
                        ) from error

            available_lengths = tuple(
                chain_length for chain_length in CHAIN_LENGTHS
                if chain_length in graphs)
            manifest["available_chain_lengths"] = list(available_lengths)
            manifest["status"] = "measuring"
            _write_json(manifest_path, manifest)
            samples_by_cell = {
                (mode, chain_length): []
                for chain_length in available_lengths
                for mode in MODES
            }
            sequence = 0
            for trial_index in range(TRIALS):
                for order, (mode, chain_length) in enumerate(
                        _ordered_cells(trial_index, available_lengths), start=1):
                    sequence += 1
                    timing = _measure(
                        stream, mode, chain_length, left, right, increment,
                        graphs[chain_length])
                    sample = {
                        "schema_version": SCHEMA_VERSION,
                        "sequence": sequence,
                        "trial": trial_index + 1,
                        "order": order,
                        "mode": mode,
                        "chain_length": chain_length,
                        **timing,
                    }
                    samples_by_cell[(mode, chain_length)].append(sample)
                    _append_json_line(trials_path, sample)

            stream.synchronize()
            validation = bool(
                torch.isfinite(left).all().item()
                and torch.isfinite(right).all().item())
            if not validation:
                raise RuntimeError("tensor validation failed")

        summary = {
            "schema_version": SCHEMA_VERSION,
            "validation": "pass",
            "cells": [
                {
                    "mode": mode,
                    "chain_length": chain_length,
                    "sample_count": len(samples_by_cell[(mode, chain_length)]),
                    "timing": _summary(
                        samples_by_cell[(mode, chain_length)]),
                }
                for chain_length in available_lengths
                for mode in MODES
            ],
        }
        _write_json(summary_path, summary)
        manifest["status"] = (
            "complete" if not manifest["graph_capture_failures"]
            else "complete_with_optional_graph_capture_failure")
        manifest["completed_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
