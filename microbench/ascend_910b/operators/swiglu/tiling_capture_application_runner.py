#!/usr/bin/env python3
"""Run a bounded batch of SwiGLU dry-launch queries in one NPU process."""

import argparse
import json
import os
from pathlib import Path
import resource

import torch
import torch_npu


CAPTURE_DIRECTORY_ENVIRONMENT = "MIDLAYER_ASCEND_SWIGLU_CAPTURE_DIR"
EXPECTED_EXECUTIONS = 6


def _positive_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _load_workloads(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "workloads"}:
        raise ValueError("SwiGLU application batch fields changed")
    if payload["schema_version"] != 1:
        raise ValueError("SwiGLU application batch schema must be 1")
    workloads = payload["workloads"]
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("SwiGLU application batch must be a non-empty list")
    normalized = []
    case_keys = set()
    capture_directories = set()
    for index, workload in enumerate(workloads):
        if not isinstance(workload, dict) or set(workload) != {
                "capture_directory", "case_key", "dtype", "intermediate",
                "tok"}:
            raise ValueError(
                f"SwiGLU application workload {index} fields changed")
        case_key = workload["case_key"]
        if not isinstance(case_key, str) or not case_key:
            raise ValueError(
                f"SwiGLU application workload {index} has invalid case key")
        if case_key in case_keys:
            raise ValueError(f"duplicate SwiGLU application case: {case_key}")
        case_keys.add(case_key)
        capture_directory = Path(workload["capture_directory"]).resolve()
        if not capture_directory.is_dir():
            raise FileNotFoundError(
                f"SwiGLU capture directory is missing: {capture_directory}")
        if capture_directory in capture_directories:
            raise ValueError(
                f"duplicate SwiGLU capture directory: {capture_directory}")
        capture_directories.add(capture_directory)
        if any(capture_directory.iterdir()):
            raise ValueError(
                f"SwiGLU capture directory must be empty: {capture_directory}")
        if workload["dtype"] != "bf16":
            raise ValueError("SwiGLU tiling capture supports only bf16")
        normalized.append({
            "case_key": case_key,
            "capture_directory": capture_directory,
            "dtype": workload["dtype"],
            "tok": _positive_integer(workload["tok"], f"workload {index} tok"),
            "intermediate": _positive_integer(
                workload["intermediate"],
                f"workload {index} intermediate"),
        })
    return normalized


def _run_workload(workload):
    os.environ[CAPTURE_DIRECTORY_ENVIRONMENT] = str(
        workload["capture_directory"])
    tensor = torch.empty(
        workload["tok"], 2 * workload["intermediate"],
        dtype=torch.bfloat16, device="npu")
    torch.npu.synchronize()
    output = None
    for _ in range(EXPECTED_EXECUTIONS):
        output = torch_npu.npu_swiglu(tensor)
    torch.npu.synchronize()
    del output
    del tensor


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-batch", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    args = parser.parse_args(argv)
    if args.device_id < 0:
        raise ValueError("device id must be non-negative")
    workloads = _load_workloads(args.workload_batch.resolve())
    torch.npu.set_device(args.device_id)
    for workload in workloads:
        _run_workload(workload)
        print(json.dumps({
            "case_key": workload["case_key"],
            "state": "dry_launch_complete",
        }, sort_keys=True), flush=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if usage.ru_maxrss <= 0:
        raise RuntimeError("SwiGLU application could not read process peak RSS")
    print(json.dumps({
        "completed_workload_count": len(workloads),
        "process_peak_rss_kib": usage.ru_maxrss,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
