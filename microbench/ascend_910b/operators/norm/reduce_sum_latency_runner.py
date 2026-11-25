#!/usr/bin/env python3
"""Run the independent 910B UB-resident ReduceSum mechanism sweep."""

import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess


LUT_COUNTS = (
    32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384,
)
# These non-geometric points validate interpolation only.  They are never
# published as HWSpec coordinates, which keeps the parameter sweep independent
# of production/evaluation hidden sizes.
INTERPOLATION_VALIDATION_COUNTS = (2560, 5120)
COUNTS = tuple(sorted(LUT_COUNTS + INTERPOLATION_VALIDATION_COUNTS))
DEFAULT_ITERATIONS = 20_000
DEFAULT_TRIALS = 3
SYSTEM_CYCLE_NS = 20.0
_ROTATION_STRIDE = 5
_OUTPUT_PATTERN = re.compile(
    r"^reduce_sum_latency: device_id=(?P<device_id>\d+) "
    r"count=(?P<count>\d+) iterations=(?P<iterations>\d+) "
    r"system_cycles=(?P<system_cycles>\d+) "
    r"per_iteration_ns=(?P<per_iteration_ns>\d+(?:\.\d+)?) "
    r"validation=(?P<validation>-?\d+(?:\.\d+)?) "
    r"validation_status=pass$")
_FIELDS = (
    "run_id", "trial", "position", "device_id", "count", "iterations",
    "system_cycles", "per_iteration_ns", "validation",
    "validation_status",
)


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _schedule(trials):
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials <= 0:
        raise ValueError("trials must be positive")
    for trial in range(trials):
        offset = (trial * _ROTATION_STRIDE) % len(COUNTS)
        for position in range(len(COUNTS)):
            yield trial, position, COUNTS[(position + offset) % len(COUNTS)]


def _parse_output(text, expected_device, expected_count, iterations):
    if not isinstance(text, str):
        raise TypeError("benchmark stdout must be text")
    match = _OUTPUT_PATTERN.fullmatch(text.strip())
    if match is None:
        raise ValueError("benchmark stdout is not one validated result line")
    values = match.groupdict()
    device_id = int(values["device_id"])
    count = int(values["count"])
    observed_iterations = int(values["iterations"])
    system_cycles = int(values["system_cycles"])
    per_iteration_ns = float(values["per_iteration_ns"])
    validation = float(values["validation"])
    if device_id != expected_device:
        raise ValueError("benchmark device identity changed")
    if count != expected_count:
        raise ValueError("benchmark count identity changed")
    if observed_iterations != iterations:
        raise ValueError("benchmark iteration count changed")
    if system_cycles <= 0:
        raise ValueError("benchmark system cycles must be positive")
    recomputed_ns = system_cycles * SYSTEM_CYCLE_NS / iterations
    if not math.isclose(
            per_iteration_ns, recomputed_ns, rel_tol=0.0, abs_tol=5.01e-4):
        raise ValueError("benchmark per-iteration time is inconsistent")
    if not math.isfinite(validation) or validation != float(count):
        raise ValueError("benchmark reduction result is invalid")
    return {
        "device_id": device_id,
        "count": count,
        "iterations": observed_iterations,
        "system_cycles": system_cycles,
        "per_iteration_ns": f"{per_iteration_ns:.3f}",
        "validation": f"{validation:.6f}",
        "validation_status": "pass",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument(
        "--iterations", type=_positive_integer, default=DEFAULT_ITERATIONS)
    parser.add_argument("--trials", type=_positive_integer, default=DEFAULT_TRIALS)
    args = parser.parse_args()

    if not args.run_id.strip():
        raise ValueError("run_id must be non-empty")
    if args.device_id < 0:
        raise ValueError("device_id must be non-negative")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("binary must be an executable file")
    output = args.output.resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError("output parent directory does not exist")

    environment = os.environ.copy()
    environment["ASCEND_DEVICE_ID"] = str(args.device_id)
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS)
        writer.writeheader()
        stream.flush()
        for trial, position, count in _schedule(args.trials):
            command = (str(binary), str(count), str(args.iterations))
            result = subprocess.run(
                command, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(
                    "ReduceSum benchmark failed: "
                    f"command={command!r}, returncode={result.returncode}, "
                    f"stdout={result.stdout!r}, stderr={result.stderr!r}")
            row = _parse_output(
                result.stdout, args.device_id, count, args.iterations)
            writer.writerow({
                "run_id": args.run_id,
                "trial": trial,
                "position": position,
                "device_id": row["device_id"],
                "count": row["count"],
                "iterations": row["iterations"],
                "system_cycles": row["system_cycles"],
                "per_iteration_ns": row["per_iteration_ns"],
                "validation": row["validation"],
                "validation_status": row["validation_status"],
            })
            stream.flush()
            print(
                f"trial={trial} position={position} count={count} "
                f"per_iteration_ns={row['per_iteration_ns']}",
                flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
