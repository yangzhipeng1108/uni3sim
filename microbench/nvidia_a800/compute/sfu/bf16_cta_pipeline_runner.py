#!/usr/bin/env python3
"""Capture and validate the independent A800 BF16 CTA pipeline probe."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from statistics import median


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output/microbench/nvidia_a800"
SOURCE_PATH = Path(__file__).with_name("bf16_cta_pipeline.cu")
RUNNER_PATH = Path(__file__).resolve()
NVCC = "/usr/local/cuda-12.2/bin/nvcc"
CUOBJDUMP = "/usr/local/cuda-12.2/bin/cuobjdump"
EXPECTED_DEVICE = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
EXPECTED_THREADS = 1024
EXPECTED_REGISTERS = {
    "Dependency0": 12,
    "Dependency1": 14,
}
EXPECTED_STATIC_SHARED_BYTES = 0
EXPECTED_LOCAL_BYTES = 0
MINIMUM_TRIALS = 3
MAXIMUM_TRIALS = 11
MAXIMUM_IDLE_MEMORY_MIB = 8
CELLS = {
    "single_cta_steady": (108, 512, "independent_rounds"),
    "dual_cta_short": (216, 64, "independent_rounds"),
    "dual_cta_steady": (216, 512, "independent_rounds"),
    "dual_cta_next_wave_steady": (324, 512, "independent_rounds"),
    "single_cta_dependent_control": (108, 512, "loop_carried_sfu"),
}
RAW_FIELDS = (
    "trial", "cell", "dependency", "grid_blocks", "block_threads",
    "rounds", "resident_cta_limit", "active_sms", "loaded_bytes",
    "stored_bytes", "minimum_ctas_per_sm", "maximum_ctas_per_sm",
    "maximum_sm_span_cycles", "median_block_cycles", "checksum",
)
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


def _run(command, *, cwd, stdout=None, stderr=None, capture_output=False):
    return subprocess.run(
        command, cwd=cwd, check=True, text=True, stdout=stdout,
        stderr=stderr, capture_output=capture_output, timeout=600)


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
        "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT, capture_output=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise ValueError("unexpected nvidia-smi device row")
        rows.append({
            "index": int(row[0]),
            "name": row[1],
            "uuid": row[2],
            "pci_bus_id": row[3],
            "memory_used_mib": int(row[4]),
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
            raise ValueError("unexpected nvidia-smi compute-process row")
        rows.append({"uuid": row[0], "pid": int(row[1]), "memory_mib": int(row[2])})
    return rows


def _require_idle_device(device):
    matching = [row for row in _device_rows() if row["index"] == device]
    if len(matching) != 1:
        raise ValueError(f"device {device} does not resolve exactly once")
    row = matching[0]
    if row["name"] != EXPECTED_DEVICE:
        raise ValueError(f"device {device} is {row['name']}, not {EXPECTED_DEVICE}")
    if row["memory_used_mib"] > MAXIMUM_IDLE_MEMORY_MIB:
        raise RuntimeError(f"device {device} has {row['memory_used_mib']} MiB allocated")
    if row["utilization_percent"] != 0:
        raise RuntimeError(f"device {device} has nonzero utilization")
    processes = [entry for entry in _compute_process_rows() if entry["uuid"] == row["uuid"]]
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


def _parse_resource_usage(text):
    resources = {}
    current = None
    for line in text.splitlines():
        match = SYMBOL_PATTERN.search(line)
        if match:
            current = match.group(1)
            continue
        if current is None or "bf16_cta_pipeline" not in current:
            continue
        registers = REGISTER_PATTERN.search(line)
        shared = SHARED_PATTERN.search(line)
        local = LOCAL_PATTERN.search(line)
        if registers and shared and local:
            resources[current] = {
                "registers_per_thread": int(registers.group(1)),
                "static_shared_bytes": int(shared.group(1)),
                "local_bytes": int(local.group(1)),
            }
    if len(resources) != 2:
        raise ValueError(f"expected two kernel specializations, found {len(resources)}")
    for symbol, values in resources.items():
        dependency = [name for name in EXPECTED_REGISTERS if name in symbol]
        if len(dependency) != 1:
            raise ValueError(f"kernel dependency specialization changed: {symbol}")
        if values != {
                "registers_per_thread": EXPECTED_REGISTERS[dependency[0]],
                "static_shared_bytes": EXPECTED_STATIC_SHARED_BYTES,
                "local_bytes": EXPECTED_LOCAL_BYTES}:
            raise ValueError(f"resource contract changed for {symbol}: {values}")
    return resources


def _validate_sass(text):
    required = ("LDG", "STG", "MUFU.EX2", "MUFU.RCP", "BAR.SYNC")
    missing = [opcode for opcode in required if opcode not in text]
    if missing:
        raise ValueError(f"SASS is missing required opcodes: {missing}")
    forbidden = ("LDL", "STL")
    present = [opcode for opcode in forbidden if opcode in text]
    if present:
        raise ValueError(f"SASS contains local-memory opcodes: {present}")
    return {"required_opcodes": list(required), "forbidden_opcodes_absent": list(forbidden)}


def _load_raw(path, trials):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("raw header changed")
        rows = list(reader)
    if len(rows) != trials * len(CELLS):
        raise ValueError("raw row count changed")
    seen = set()
    normalized = []
    for row in rows:
        cell = row["cell"]
        if cell not in CELLS:
            raise ValueError(f"unknown cell {cell}")
        grid, rounds, dependency = CELLS[cell]
        trial = int(row["trial"])
        key = (cell, trial)
        if key in seen or trial not in range(trials):
            raise ValueError(f"invalid cell/trial key {key}")
        seen.add(key)
        integer_fields = {
            name: int(row[name]) for name in RAW_FIELDS
            if name not in ("cell", "dependency")
        }
        if row["dependency"] != dependency:
            raise ValueError(f"dependency changed for {cell}")
        if integer_fields["grid_blocks"] != grid or integer_fields["rounds"] != rounds:
            raise ValueError(f"geometry changed for {cell}")
        if integer_fields["block_threads"] != EXPECTED_THREADS:
            raise ValueError(f"thread contract changed for {cell}")
        if integer_fields["resident_cta_limit"] != 2:
            raise ValueError(f"dual-CTA admission failed for {cell}")
        if integer_fields["active_sms"] != EXPECTED_SM_COUNT:
            raise ValueError(f"not all SMs were active for {cell}")
        expected_ctas_per_sm = grid // EXPECTED_SM_COUNT
        if (grid % EXPECTED_SM_COUNT
                or integer_fields["minimum_ctas_per_sm"]
                != expected_ctas_per_sm
                or integer_fields["maximum_ctas_per_sm"]
                != expected_ctas_per_sm):
            raise ValueError(
                f"cell {cell} does not assign its exact CTA cohort uniformly")
        thread_rounds = grid * EXPECTED_THREADS * rounds
        if integer_fields["loaded_bytes"] != 4 * thread_rounds:
            raise ValueError(f"load-byte contract changed for {cell}")
        if integer_fields["stored_bytes"] != 2 * thread_rounds:
            raise ValueError(f"store-byte contract changed for {cell}")
        if integer_fields["maximum_sm_span_cycles"] <= 0 or integer_fields["median_block_cycles"] <= 0:
            raise ValueError(f"cycle contract failed for {cell}")
        normalized.append({"cell": cell, "trial": trial, **integer_fields})
    return normalized


def _derive_candidate(rows):
    by_cell = {}
    for cell in CELLS:
        by_cell[cell] = [row for row in rows if row["cell"] == cell]
    summaries = {}
    for cell, records in by_cell.items():
        cycles = [record["maximum_sm_span_cycles"] for record in records]
        rates = [
            (record["loaded_bytes"] + record["stored_bytes"]) / record["maximum_sm_span_cycles"]
            for record in records
        ]
        summaries[cell] = {
            "median_maximum_sm_span_cycles": median(cycles),
            "maximum_sm_span_cycle_range_ratio": (max(cycles) - min(cycles)) / median(cycles),
            "median_bytes_per_sm_cycle": median(rates) / EXPECTED_SM_COUNT,
        }
    single_rate = summaries["single_cta_steady"]["median_bytes_per_sm_cycle"]
    dual_rate = summaries["dual_cta_steady"]["median_bytes_per_sm_cycle"]
    next_wave_rate = summaries[
        "dual_cta_next_wave_steady"]["median_bytes_per_sm_cycle"]
    dependent_rate = summaries["single_cta_dependent_control"]["median_bytes_per_sm_cycle"]
    stable = all(
        summary["maximum_sm_span_cycle_range_ratio"] <= 0.10
        for summary in summaries.values())
    return {
        "schema": "nvidia_a800_bf16_cta_pipeline_candidate_v2",
        "hardware_id": "nvidia_a800",
        "mechanism": "bf16_load_independent_sfu_store_cross_cta_handoff",
        "cell_summaries": summaries,
        "all_cells_three_or_more_trials_stable_within_10_percent": stable,
        "dual_to_single_steady_state_throughput_ratio": dual_rate / single_rate,
        "next_wave_to_dual_steady_state_throughput_ratio": (
            next_wave_rate / dual_rate),
        "independent_to_loop_carried_single_cta_throughput_ratio": single_rate / dependent_rate,
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "For this independent 1024-thread SM80 mechanism probe, the measured ratios quantify "
            "whether a second resident CTA hands BF16 global-load, independent EX2/RCP, and global-store "
            "rounds across pipeline stages more effectively than a single CTA and how one following "
            "full-SM cohort changes that aggregate mechanism throughput."),
        "excluded_conclusions": [
            "production SiLU kernel duration",
            "a production-specific correction factor",
            "a per-shape latency lookup",
            "direct composition with production stage services",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    if args.device < 0:
        raise ValueError("device must be nonnegative")
    if args.trials < MINIMUM_TRIALS or args.trials > MAXIMUM_TRIALS:
        raise ValueError(f"trials must be in [{MINIMUM_TRIALS}, {MAXIMUM_TRIALS}]")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", args.owner):
        raise ValueError("owner must match [A-Za-z0-9._-]{1,64}")
    output = _validated_output(args.output)
    device = _require_idle_device(args.device)
    output.mkdir(parents=True)
    for directory in (output / "build", output / "source_snapshot", output / "device"):
        directory.mkdir()
    source_snapshot = output / "source_snapshot" / SOURCE_PATH.name
    runner_snapshot = output / "source_snapshot" / RUNNER_PATH.name
    shutil.copy2(SOURCE_PATH, source_snapshot)
    shutil.copy2(RUNNER_PATH, runner_snapshot)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                  capture_output=True).stdout.strip()
    status = _run(["git", "status", "--short"], cwd=REPOSITORY_ROOT,
                  capture_output=True).stdout
    (output / "baseline_commit.txt").write_text(commit + "\n")
    (output / "tracked_worktree_status.txt").write_text(status)
    binary = output / "build" / "bf16_cta_pipeline"
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
    (output / "build/bf16_cta_pipeline.sass").write_text(sass_result.stdout)
    sass_contract = _validate_sass(sass_result.stdout)
    run_command = [str(binary), str(args.device), str(args.trials)]
    with (output / "raw.csv").open("w") as stdout, (
            output / "native.stderr").open("w") as stderr:
        _run(run_command, cwd=REPOSITORY_ROOT,
             stdout=stdout, stderr=stderr)
    rows = _load_raw(output / "raw.csv", args.trials)
    candidate = _derive_candidate(rows)
    (output / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    postflight = _wait_for_idle_device(args.device)
    provenance = {
        "schema": "nvidia_a800_bf16_cta_pipeline_capture_v2",
        "owner": args.owner,
        "pid": os.getpid(),
        "baseline_commit": commit,
        "command": shlex.join([sys.executable, *sys.argv]),
        "native_command": shlex.join(run_command),
        "device": device,
        "postflight_device": postflight,
        "trials": args.trials,
        "source_sha256": _sha256(source_snapshot),
        "runner_sha256": _sha256(runner_snapshot),
        "binary_sha256": _sha256(binary),
        "resources": resources,
        "sass_contract": sass_contract,
        "production_or_evaluation_duration_consumed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
