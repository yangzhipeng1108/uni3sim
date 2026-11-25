#!/usr/bin/env python3
"""Capture the source-only CANN SwiGLU tile-address control probe.

The probe has no tensor traffic, TPipe, queue, vector instruction, or
production SwiGLU call.  Each launch measures a source-equivalent dynamic
``CaclOneTileParam`` body (mode 1) against its already-modeled dynamic
quotient/remainder prefix (mode 0).  The paired difference is therefore a
mechanism observation only; it is not an operator-duration fit.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


CANN_SOURCE = (
    "/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/impl/"
    "ops_nn/ascendc/swi_glu/glu_tiling_kernel.hpp")
CANN_SOURCE_SHA256 = (
    "2b6abc813b6e372b0d36e81f79fd618c180b0c0e4de2643056fd7aad775c2053")
WORKER_COUNTS = (1, 24, 48)
BRANCHES = (1, 2, 3, 4)
MODES = (1, 2)
TRIAL_COUNT = 3
REPETITIONS = 16_384
CSV_FIELD_COUNT = 9
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
SOURCE_FILES = (
    "CMakeLists.txt",
    "swiglu_tile_address_control_kernel.cpp",
    "swiglu_tile_address_control_main.cpp",
    "tile_address_control_runner.py",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing probe source: {path}")
        payload = path.read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _run_checked(command, *, timeout):
    result = subprocess.run(
        command, check=False, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"command returned {result.returncode}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result.stdout


def _remote_command(args, command, *, timeout=60):
    return _run_checked(["ssh", args.remote_host, *command], timeout=timeout)


def _container_command(args, command, *, timeout=60):
    return _remote_command(
        args,
        [
            "docker", "exec", "-e", "ASCEND_DEVICE_ID=0", "-e",
            f"ASCEND_RT_VISIBLE_DEVICES={args.physical_device_id}",
            args.container, *command,
        ],
        timeout=timeout)


def _require_idle_device(args):
    status = _remote_command(args, ["npu-smi", "info"])
    row = re.findall(
        rf"^\|\s*{args.physical_device_id}\s+910B2\s+\|\s*(OK)\s+\|",
        status, flags=re.MULTILINE)
    if row != ["OK"]:
        raise RuntimeError(
            "selected physical device does not appear exactly once as healthy")
    if "No running processes found" not in status:
        raise RuntimeError("NPU inventory reports a running process")
    return status


def _remote_sha256(args, path):
    output = _container_command(args, ["sha256sum", path]).strip().split()
    if len(output) != 2 or re.fullmatch(r"[0-9a-f]{64}", output[0]) is None:
        raise ValueError(f"invalid SHA256 output for {path!r}")
    return output[0]


def _cells():
    return tuple(
        (workers, branch, mode)
        for workers in WORKER_COUNTS
        for branch in BRANCHES
        for mode in MODES)


def _ordered_cells(trial):
    cells = _cells()
    if trial == 0:
        return "forward", cells
    if trial == 1:
        return "reverse", tuple(reversed(cells))
    if trial == 2:
        # A rotation prevents either branch family from always following a
        # particular worker population without changing any physical cell.
        return "rotated", cells[7:] + cells[:7]
    raise ValueError("trial is outside the frozen capture plan")


def _parse_rows(output, expected_cells):
    parsed = []
    for fields in csv.reader(output.splitlines()):
        if not fields:
            continue
        if len(fields) != CSV_FIELD_COUNT:
            raise ValueError(f"unexpected child CSV row: {fields}")
        workers, worker, branch0, mode0, repetitions, cycles, checksum = (
            int(value) for value in fields[:7])
        elapsed_ns = float(fields[7])
        if fields[8] != "pass" or workers not in WORKER_COUNTS:
            raise ValueError(f"invalid child validation row: {fields}")
        if worker < 0 or worker >= workers or branch0 not in range(4):
            raise ValueError(f"child row has an invalid source index: {fields}")
        if mode0 not in range(2) or repetitions != REPETITIONS:
            raise ValueError(f"child row has an invalid mode/repetition: {fields}")
        if cycles <= 0 or checksum <= 0 or elapsed_ns <= 0.0:
            raise ValueError(f"child row has an invalid measurement: {fields}")
        parsed.append({
            "workers": workers,
            "worker": worker,
            "branch": branch0 + 1,
            "mode": mode0 + 1,
            "repetitions": repetitions,
            "cycles": cycles,
            "device_ns": elapsed_ns,
            "checksum": checksum,
            "validation": fields[8],
        })
    expected_rows = sum(cell[0] for cell in expected_cells)
    if len(parsed) != expected_rows:
        raise ValueError(
            f"child emitted {len(parsed)} rows, expected {expected_rows}")
    expected = {
        (workers, branch, mode, worker)
        for workers, branch, mode in expected_cells
        for worker in range(workers)
    }
    observed = {
        (row["workers"], row["branch"], row["mode"], row["worker"])
        for row in parsed
    }
    if observed != expected:
        raise ValueError("child output does not cover the requested cell matrix")
    return parsed


def _write_json_exclusive(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _median(values):
    if not values:
        raise ValueError("median needs at least one value")
    return statistics.median(values)


def _format_ns(value):
    return f"{value:.6f}"


def _report(rows, metadata):
    paired = {}
    for row in rows:
        key = (row["trial"], row["workers"], row["branch"], row["worker"])
        modes = paired.setdefault(key, {})
        if row["mode"] in modes:
            raise ValueError("duplicate control mode within a paired sample")
        modes[row["mode"]] = row
    samples = {}
    for key, modes in paired.items():
        if tuple(sorted(modes)) != MODES:
            raise ValueError("missing paired source control mode")
        baseline = modes[1]
        address = modes[2]
        delta_cycles = address["cycles"] - baseline["cycles"]
        if delta_cycles <= 0:
            raise RuntimeError("address control did not exceed shared div/rem")
        sample_key = (key[1], key[2])
        samples.setdefault(sample_key, []).append(
            delta_cycles * 20.0 / REPETITIONS)
    lines = [
        "# 910B CANN SwiGLU tile-address source-control capture",
        "",
        "This is an independent mechanism probe, not a production kernel or Serving duration measurement.",
        "",
        f"- Run: `{metadata['run_id']}`",
        f"- Source commit: `{metadata['source_commit']}`",
        f"- Probe snapshot SHA256: `{metadata['probe_snapshot_sha256']}`",
        f"- Probe binary SHA256: `{metadata['binary_sha256']}`",
        f"- CANN source: `{CANN_SOURCE}`",
        f"- CANN source SHA256: `{metadata['cann_source_sha256']}`",
        f"- Physical NPU: `{metadata['physical_device_id']}` (idle before and after capture)",
        f"- SYS_CNT tick: `20 ns`; repetitions per worker: `{REPETITIONS}`",
        "",
        "Mode 0 executes CaclOneTileParam's dynamic quotient/remainder only. Mode 1 executes the same quotient/remainder plus the exact source GM-offset arithmetic and base/tail selection. The reported difference isolates the latter, so it cannot re-price the existing div/rem service.",
        "",
        "| AIV workers | source branch | paired samples | median extra ns/tile | min | max |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    branch_names = {
        1: "base row/base col",
        2: "base row/tail col",
        3: "tail row/base col",
        4: "tail row/tail col",
    }
    for workers in WORKER_COUNTS:
        for branch in BRANCHES:
            values = samples[(workers, branch)]
            lines.append(
                f"| {workers} | {branch_names[branch]} | {len(values)} | "
                f"{_format_ns(_median(values))} | {_format_ns(min(values))} | "
                f"{_format_ns(max(values))} |")
    lines.extend((
        "",
        "The capture is evidence only. No HWSpec service or production predictor is changed by this runner; a service requires separate review of scope, worker scheduling, and version binding.",
        "",
    ))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--physical-device-id", required=True, type=int)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("run ID must match [A-Za-z0-9._-]{1,96}")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        parser.error("source commit must be a 40-character lowercase SHA")
    if args.physical_device_id < 0 or args.physical_device_id > 7:
        parser.error("physical device ID must be in [0, 7]")
    if args.output_json.exists() or args.report.exists():
        parser.error("output paths must not already exist")
    if args.output_json.parent != Path("profiler_output") and (
            Path("profiler_output") not in args.output_json.parents):
        parser.error("raw JSON must be stored under profiler_output")
    if args.report.parent != Path("experiment_logs") and (
            Path("experiment_logs") not in args.report.parents):
        parser.error("report must be stored under experiment_logs")

    benchmark_root = Path(__file__).resolve().parent
    probe_snapshot = _snapshot_sha256(benchmark_root)
    _require_idle_device(args)
    cann_sha = _remote_sha256(args, CANN_SOURCE)
    if cann_sha != CANN_SOURCE_SHA256:
        raise RuntimeError("CANN SwiGLU source identity does not match 8.5.1")
    binary_sha = _remote_sha256(args, args.binary)
    rows = []
    for trial in range(TRIAL_COUNT):
        direction, ordered_cells = _ordered_cells(trial)
        batch = ["--batch"]
        for workers, branch, mode in ordered_cells:
            batch.append(f"{workers}:{branch}:{mode}:{REPETITIONS}")
        output = _container_command(args, [args.binary, *batch], timeout=120)
        parsed_rows = _parse_rows(output, ordered_cells)
        for row in parsed_rows:
            row["run_id"] = args.run_id
            row["trial"] = trial + 1
            row["order_direction"] = direction
            rows.append(row)
        _require_idle_device(args)
        print(
            f"trial={trial + 1}/{TRIAL_COUNT} rows={len(parsed_rows)} "
            f"order={direction}",
            file=sys.stderr, flush=True)
    metadata = {
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "probe_snapshot_sha256": probe_snapshot,
        "binary_sha256": binary_sha,
        "cann_source_path": CANN_SOURCE,
        "cann_source_sha256": cann_sha,
        "physical_device_id": args.physical_device_id,
        "counter_tick_ns": 20.0,
        "repetitions": REPETITIONS,
        "trials": TRIAL_COUNT,
        "cell_count": len(_cells()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(args.output_json, {
        "metadata": metadata,
        "rows": rows,
    })
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(_report(rows, metadata))
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
