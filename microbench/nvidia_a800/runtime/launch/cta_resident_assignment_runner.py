#!/usr/bin/env python3
"""Capture and validate the A800 eight-resident-CTA assignment probe."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output/microbench/nvidia_a800"
SOURCE_PATH = Path(__file__).with_name("cta_resident_assignment.cu")
RUNNER_PATH = Path(__file__).resolve()
NVCC = "/usr/local/cuda-12.2/bin/nvcc"
CUOBJDUMP = "/usr/local/cuda-12.2/bin/cuobjdump"
EXPECTED_DEVICE = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
EXPECTED_BLOCK_THREADS = 256
EXPECTED_RESIDENT_CTAS_PER_SM = 8
EXPECTED_STATIC_SHARED_BYTES = 16
RESIDENT_COHORT = EXPECTED_SM_COUNT * EXPECTED_RESIDENT_CTAS_PER_SM
GRID_SIZES = (1, 108, 864, 1024)
MINIMUM_TRIALS = 3
MAXIMUM_TRIALS = 11
MAXIMUM_IDLE_MEMORY_MIB = 8
RAW_FIELDS = (
    "grid_size", "trial", "block_index", "sm_id", "arrival_ticket",
    "release_ticket", "start_cycle", "end_cycle",
)
META_FIELDS = {
    "schema", "device_ordinal", "device_name", "device_uuid", "pci_bus_id",
    "compute_capability", "sm_count", "block_threads",
    "resident_ctas_per_sm", "warmup_launches",
    "production_or_evaluation_duration_consumed",
}
SYMBOL_PATTERN = re.compile(r"Function(?:\s*:\s*|\s+)(\S+?):?$")
REGISTER_PATTERN = re.compile(r"REG:(\d+)")
SHARED_PATTERN = re.compile(r"SHARED:(\d+)")
LOCAL_PATTERN = re.compile(r"LOCAL:(\d+)")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd, env=None, stdout=None, stderr=None,
         capture_output=False):
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, text=True, stdout=stdout,
        stderr=stderr, capture_output=capture_output, timeout=600)


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _validated_output(path):
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be below {root}")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def _device_rows():
    query = "index,name,uuid,pci.bus_id,memory.used,utilization.gpu"
    result = _run([
        "nvidia-smi", f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT, capture_output=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise ValueError("unexpected nvidia-smi device row")
        rows.append({
            "index": int(row[0]), "name": row[1], "uuid": row[2],
            "pci_bus_id": row[3], "memory_used_mib": int(row[4]),
            "utilization_percent": int(row[5]),
        })
    return rows


def _compute_process_rows():
    result = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT, capture_output=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("unexpected nvidia-smi process row")
        rows.append({
            "uuid": row[0], "pid": int(row[1]), "memory_mib": int(row[2])})
    return rows


def _require_idle_device(device):
    matching = [row for row in _device_rows() if row["index"] == device]
    if len(matching) != 1:
        raise ValueError(f"device {device} does not resolve exactly once")
    row = matching[0]
    if row["name"] != EXPECTED_DEVICE:
        raise ValueError(f"device {device} is not {EXPECTED_DEVICE}")
    if (row["memory_used_mib"] > MAXIMUM_IDLE_MEMORY_MIB
            or row["utilization_percent"] != 0):
        raise RuntimeError(f"device {device} is not idle: {row}")
    processes = [
        process for process in _compute_process_rows()
        if process["uuid"] == row["uuid"]]
    if processes:
        raise RuntimeError(f"device {device} has compute processes: {processes}")
    return row


def _wait_for_idle_device(device):
    last_error = None
    for _ in range(10):
        try:
            return _require_idle_device(device)
        except RuntimeError as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"device {device} did not return idle: {last_error}")


def parse_native_output(text, expected_trials):
    if not isinstance(text, str) or not text:
        raise ValueError("native output must be non-empty text")
    meta = {}
    raw_header = None
    records = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        columns = tuple(next(csv.reader([line])))
        if columns[0] == "meta":
            if len(columns) != 3 or columns[1] in meta:
                raise ValueError(f"invalid native meta row {line_number}")
            meta[columns[1]] = columns[2]
        elif columns[0] == "raw" and columns[1:] == RAW_FIELDS:
            if raw_header is not None:
                raise ValueError("native output repeats raw header")
            raw_header = columns[1:]
        elif columns[0] == "raw":
            if len(columns) != len(RAW_FIELDS) + 1:
                raise ValueError(f"invalid raw row {line_number}")
            values = {
                name: int(value)
                for name, value in zip(RAW_FIELDS, columns[1:])}
            key = (
                values["grid_size"], values["trial"],
                values["block_index"])
            if key in records:
                raise ValueError(f"duplicate raw record {key}")
            records[key] = values
        else:
            raise ValueError(f"unknown native row {line_number}")
    if set(meta) != META_FIELDS or raw_header != RAW_FIELDS:
        raise ValueError("native metadata/header schema changed")
    expected_meta = {
        "schema": "nvidia_a800_cta_resident_assignment_native_v1",
        "device_ordinal": "0", "device_name": EXPECTED_DEVICE,
        "compute_capability": "sm_80", "sm_count": str(EXPECTED_SM_COUNT),
        "block_threads": str(EXPECTED_BLOCK_THREADS),
        "resident_ctas_per_sm": str(EXPECTED_RESIDENT_CTAS_PER_SM),
        "warmup_launches": "3",
        "production_or_evaluation_duration_consumed": "false",
    }
    for name, expected in expected_meta.items():
        if meta[name] != expected:
            raise ValueError(
                f"native metadata {name} changed: {meta[name]!r}")
    if re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            meta["device_uuid"]) is None:
        raise ValueError("native UUID is malformed")
    expected_keys = {
        (grid, trial, block)
        for grid in GRID_SIZES
        for trial in range(expected_trials)
        for block in range(grid)}
    if set(records) != expected_keys:
        raise ValueError("native raw matrix is incomplete")
    for grid in GRID_SIZES:
        for trial in range(expected_trials):
            cell = [records[(grid, trial, block)] for block in range(grid)]
            if {row["arrival_ticket"] for row in cell} != set(range(grid)):
                raise ValueError("arrival tickets are not a permutation")
            if {row["release_ticket"] for row in cell} != set(range(grid)):
                raise ValueError("release tickets are not a permutation")
            if any(
                    row["sm_id"] not in range(EXPECTED_SM_COUNT)
                    or row["end_cycle"] < row["start_cycle"]
                    for row in cell):
                raise ValueError("native CTA record is outside bounds")
    return {"meta": meta, "records": records}


def analyze_assignment(native, raw_sha256):
    records = native["records"]
    cells = {}
    all_partial_mechanism_contracts_passed = True
    tail_signatures = []
    tail_nonuniform_observed = False
    trials = max(key[1] for key in records) + 1
    for grid in GRID_SIZES:
        resident_target = min(grid, RESIDENT_COHORT)
        for trial in range(trials):
            rows = [records[(grid, trial, block)] for block in range(grid)]
            total = [0] * EXPECTED_SM_COUNT
            resident = [0] * EXPECTED_SM_COUNT
            tail = [0] * EXPECTED_SM_COUNT
            for row in rows:
                sm_id = row["sm_id"]
                total[sm_id] += 1
                if row["arrival_ticket"] < resident_target:
                    resident[sm_id] += 1
                else:
                    tail[sm_id] += 1
            if grid == 1:
                passed = max(total) == 1 and sum(total) == 1
            elif grid == 108:
                passed = min(total) == max(total) == 1
            elif grid == 864:
                passed = min(resident) == max(resident) == 8
            elif grid == 1024:
                passed = (
                    min(resident) == max(resident) == 8
                    and max(math.ceil(value / 8) for value in total) == 2)
                tail_signatures.append(tuple(tail))
                tail_nonuniform_observed = (
                    tail_nonuniform_observed or min(tail) != max(tail))
            else:
                raise AssertionError(grid)
            cells[f"grid{grid}:trial{trial}"] = {
                "observed_sm_count": sum(value > 0 for value in total),
                "minimum_total_ctas_per_sm": min(total),
                "maximum_total_ctas_per_sm": max(total),
                "minimum_resident_cohort_ctas_per_sm": min(resident),
                "maximum_resident_cohort_ctas_per_sm": max(resident),
                "minimum_queued_tail_ctas_per_sm": min(tail),
                "maximum_queued_tail_ctas_per_sm": max(tail),
                "maximum_assignment_waves": max(
                    math.ceil(value / 8) for value in total),
                "partial_mechanism_contract_passed": passed,
                "tail_exact_assignment_publishable": False,
            }
            all_partial_mechanism_contracts_passed = (
                all_partial_mechanism_contracts_passed and passed)
    tail_exact_assignment_stable = all(
        signature == tail_signatures[0]
        for signature in tail_signatures[1:])
    return {
        "schema": "nvidia_a800_cta_resident_assignment_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "eight_resident_256_thread_cta_assignment",
        "exact_grid_sizes": list(GRID_SIZES),
        "trial_count": trials,
        "resident_ctas_per_sm": EXPECTED_RESIDENT_CTAS_PER_SM,
        "cell_analysis": cells,
        "all_partial_mechanism_contracts_passed": (
            all_partial_mechanism_contracts_passed),
        "tail_nonuniform_observed": tail_nonuniform_observed,
        "cross_trial_exact_tail_assignment_stable": (
            tail_exact_assignment_stable),
        "tail_exact_assignment_publishable": False,
        "evidence_status": (
            "partial_mechanism_observation"
            if all_partial_mechanism_contracts_passed
            else "evidence_pending"),
        "raw_sha256": raw_sha256,
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "For this independent 256-thread, eight-resident-CTA SM80 probe, "
            "grid 108 assigns one CTA per SM, the first 864 admitted CTAs "
            "assign eight per SM at grids 864 and 1024, and grid 1024 has "
            "two assignment waves. The nonuniform, cross-trial-variable tail "
            "falsifies an exact even 9/10-CTA-per-SM assignment."),
        "excluded_conclusions": [
            "production AddRMSNorm duration",
            "a task-floor extrapolation beyond its measured launch domain",
            "deterministic CTA-to-SM identity for a different source program",
            "the exact queued-tail assignment or any tail average as a "
            "Forward parameter",
        ],
    }


def _parse_resource_usage(text):
    current = None
    matches = []
    for line in text.splitlines():
        symbol = SYMBOL_PATTERN.search(line)
        if symbol:
            current = symbol.group(1)
            continue
        if current is None or "cta_resident_assignment_probe" not in current:
            continue
        registers = REGISTER_PATTERN.search(line)
        shared = SHARED_PATTERN.search(line)
        local = LOCAL_PATTERN.search(line)
        if registers and shared and local:
            matches.append({
                "symbol": current,
                "registers_per_thread": int(registers.group(1)),
                "static_shared_bytes": int(shared.group(1)),
                "local_bytes": int(local.group(1)),
            })
    if len(matches) != 1:
        raise ValueError("expected one resident-assignment kernel resource row")
    usage = matches[0]
    if (usage["registers_per_thread"] > 32
            or usage["static_shared_bytes"] != EXPECTED_STATIC_SHARED_BYTES
            or usage["local_bytes"] != 0):
        raise ValueError(f"kernel resource contract prevents 8 CTA/SM: {usage}")
    return usage


def _validate_sass(text):
    required_groups = (("S2R", "CS2R"), ("ATOMG", "RED.E"), ("BAR.SYNC",))
    if any(not any(opcode in text for opcode in group)
           for group in required_groups):
        raise ValueError("SASS lacks SMID/atomic/barrier mechanism opcodes")
    if "LDL" in text or "STL" in text:
        raise ValueError("SASS contains local-memory operations")
    return {
        "smid_read_present": True,
        "global_atomic_present": True,
        "cta_barrier_present": True,
        "local_memory_opcodes_absent": True,
    }


def capture(arguments):
    output = _validated_output(arguments.output)
    device = _require_idle_device(arguments.device)
    output.mkdir(parents=True)
    for directory in (output / "build", output / "source_snapshot"):
        directory.mkdir()
    source_snapshot = output / "source_snapshot" / SOURCE_PATH.name
    runner_snapshot = output / "source_snapshot" / RUNNER_PATH.name
    shutil.copy2(SOURCE_PATH, source_snapshot)
    shutil.copy2(RUNNER_PATH, runner_snapshot)
    commit = _run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout.strip()
    status = _run(
        ["git", "status", "--short"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout
    (output / "baseline_commit.txt").write_text(commit + "\n")
    (output / "tracked_worktree_status.txt").write_text(status)
    binary = output / "build" / "cta_resident_assignment"
    compile_command = [
        NVCC, "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo",
        "-Xptxas=-v", str(source_snapshot), "-o", str(binary),
    ]
    with (output / "build/compile.stdout").open("w") as stdout, (
            output / "build/compile.stderr").open("w") as stderr:
        _run(compile_command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    resource_result = _run(
        [CUOBJDUMP, "--dump-resource-usage", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True)
    (output / "build/resource_usage.txt").write_text(resource_result.stdout)
    resources = _parse_resource_usage(resource_result.stdout)
    sass_result = _run(
        [CUOBJDUMP, "--dump-sass", str(binary)], cwd=REPOSITORY_ROOT,
        capture_output=True)
    (output / "build/cta_resident_assignment.sass").write_text(
        sass_result.stdout)
    sass_contract = _validate_sass(sass_result.stdout)
    raw_path = output / "raw.csv"
    native_command = [str(binary), "--trials", str(arguments.trials)]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device)
    with raw_path.open("w") as stdout, (
            output / "native.stderr").open("w") as stderr:
        _run(native_command, cwd=REPOSITORY_ROOT, env=environment,
             stdout=stdout, stderr=stderr)
    native = parse_native_output(
        raw_path.read_text(encoding="utf-8"), arguments.trials)
    if native["meta"]["device_uuid"] != device["uuid"]:
        raise ValueError("native UUID disagrees with selected physical device")
    candidate = analyze_assignment(native, _sha256(raw_path))
    _write_json(output / "candidate.json", candidate)
    postflight = _wait_for_idle_device(arguments.device)
    provenance = {
        "schema": "nvidia_a800_cta_resident_assignment_capture_v1",
        "hardware_id": "nvidia_a800",
        "owner": arguments.owner,
        "pid": os.getpid(),
        "baseline_commit": commit,
        "command": shlex.join([sys.executable, *sys.argv]),
        "native_command": shlex.join(native_command),
        "device": device,
        "postflight_device": postflight,
        "trials": arguments.trials,
        "source_sha256": _sha256(source_snapshot),
        "runner_sha256": _sha256(runner_snapshot),
        "binary_sha256": _sha256(binary),
        "raw_sha256": _sha256(raw_path),
        "resources": resources,
        "sass_contract": sass_contract,
        "production_or_evaluation_duration_consumed": False,
    }
    _write_json(output / "provenance.json", provenance)
    return {
        "output": str(output),
        "device": device["index"],
        "evidence_status": candidate["evidence_status"],
        "candidate_sha256": _sha256(output / "candidate.json"),
        "provenance_sha256": _sha256(output / "provenance.json"),
    }


def _arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--owner", required=True)
    arguments = parser.parse_args(argv)
    if arguments.device < 0:
        raise ValueError("device must be nonnegative")
    if arguments.trials < MINIMUM_TRIALS or arguments.trials > MAXIMUM_TRIALS:
        raise ValueError(
            f"trials must be in [{MINIMUM_TRIALS}, {MAXIMUM_TRIALS}]")
    if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", arguments.owner) is None:
        raise ValueError("owner must match [A-Za-z0-9._-]{1,64}")
    return arguments


def main(argv=None):
    result = capture(_arguments(argv))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
