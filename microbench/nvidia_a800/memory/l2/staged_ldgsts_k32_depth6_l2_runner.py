#!/usr/bin/env python3
"""Build and capture the independent SM80 K32 depth-6 LDGSTS probe."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time

from staged_ldgsts_l2_runner import (
    CUOBJDUMP,
    NVCC,
    OUTPUT_ROOT,
    _capture,
    _device,
    _sha256,
    _wait_for_idle,
)


SOURCE = Path(__file__).with_name("staged_ldgsts_k32_depth6_l2.cu")
LOGICAL_BYTES = 216 * 4096 * (12 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.trials < 3 or args.trials > 11:
        raise ValueError("trials must be in [3, 11]")
    output = args.output_dir.resolve()
    if output == OUTPUT_ROOT.resolve() or OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("output-dir must be an exclusive A800 profiler child")
    if output.exists():
        raise FileExistsError(output)
    device_before = _device(args.device)
    output.mkdir(parents=True)
    binary = output / "staged_ldgsts_k32_depth6_l2"
    compile_command = [
        str(NVCC), "-std=c++17", "-O3",
        "--generate-code=arch=compute_80,code=sm_80", "--disable-warnings",
        "-o", str(binary), str(SOURCE),
    ]
    _capture(compile_command)
    sass = _capture([str(CUOBJDUMP), "--dump-sass", str(binary)]).stdout
    if "LDGSTS" not in sass or "LDL" in sass or "STL" in sass:
        raise ValueError("compiled SASS violates the LDGSTS contract")
    (output / "probe.sass.txt").write_text(sass, encoding="utf-8")
    environment = dict(os.environ)
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if name in environment:
            raise RuntimeError(f"{name} must be unset")
    command = [str(binary), str(args.device), str(args.trials)]
    started = time.time()
    result = _capture(command, environment=environment)
    completed = time.time()
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != args.trials:
        raise ValueError("probe trial count changed")
    durations = []
    for trial, row in enumerate(rows):
        if (int(row["trial"]) != trial
                or int(row["logical_bytes"]) != LOGICAL_BYTES
                or int(row["resident_ctas_per_sm"]) != 2):
            raise ValueError("probe row violates the frozen contract")
        duration_ms = float(row["duration_ms"])
        if not math.isfinite(duration_ms) or duration_ms <= 0.0:
            raise ValueError("probe duration must be finite and positive")
        durations.append(duration_ms)
    center = statistics.median(durations)
    spread = (max(durations) - min(durations)) / center
    if spread > 0.10:
        raise ValueError(f"probe stability gate failed: {spread}")
    (output / "raw.csv").write_text(result.stdout, encoding="utf-8")
    device_after = _wait_for_idle(args.device)
    provenance = {
        "schema": "nvidia_a800_staged_ldgsts_k32_depth6_l2_service_v1",
        "hardware": "nvidia_a800",
        "owner": "kernel-accuracy-gemm",
        "device_before": device_before,
        "device_after": device_after,
        "baseline_commit": "655f609e3bf61491757dcc04c01714ced27299a6",
        "source_sha256": _sha256(SOURCE),
        "binary_sha256": _sha256(binary),
        "sass_sha256": _sha256(output / "probe.sass.txt"),
        "compile_command": compile_command,
        "command": command,
        "pid": os.getpid(),
        "started_at_unix_s": started,
        "completed_at_unix_s": completed,
        "trials": args.trials,
        "logical_bytes_per_trial": LOGICAL_BYTES,
        "duration_ms": durations,
        "relative_range": spread,
        "logical_byte_capacity_per_s": LOGICAL_BYTES / (center * 1e-3),
        "production_or_evaluation_duration_consumed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
