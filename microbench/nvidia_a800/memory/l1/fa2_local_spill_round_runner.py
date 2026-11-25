#!/usr/bin/env python3
"""Capture the independent SM80 FA2 local-stack spill-round service."""

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
from statistics import median


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output/microbench/nvidia_a800"
SOURCE_PATH = Path(__file__).with_name("fa2_local_spill_round.cu")
RUNNER_PATH = Path(__file__).resolve()
NVCC = "/usr/local/cuda-12.2/bin/nvcc"
CUOBJDUMP = "/usr/local/cuda-12.2/bin/cuobjdump"
EXPECTED_DEVICE = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
EXPECTED_PRODUCTION_BINARY_SHA256 = (
    "7c68f17400fac3f69c079842bd5304d3590e28dd5da850ff1465ff7ce86a1e93")
PRODUCTION_CONTRACTS = {
    "d64_chunked": {
        "symbol": (
            "_ZN5flash24flash_fwd_splitkv_kernelI23Flash_fwd_kernel_traits"
            "ILi64ELi64ELi256ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_"
            "kernel_traitsILi64ELi64ELi256ELi4ES3_EELb1ELb0ELb0ELb0ELb1E"
            "Lb0ELb0ELb0EEEvNS_16Flash_fwd_paramsE"),
        "stack_bytes_per_thread": 56,
        "scalar_load_sites": 13,
        "scalar_store_sites": 13,
        "artifact_name": "production_d64_chunked",
        "mechanism": (
            "sm80_128_thread_cta_56_byte_stack_13_scalar_store_"
            "13_scalar_load_round"),
    },
    "d128_causal": {
        "symbol": (
            "_ZN5flash16flash_fwd_kernelI23Flash_fwd_kernel_traitsILi128E"
            "Li128ELi64ELi4ELb0ELb0EN7cutlass10bfloat16_tE19Flash_kernel_"
            "traitsILi128ELi128ELi64ELi4ES3_EELb0ELb1ELb0ELb0ELb1ELb1E"
            "Lb0ELb0EEEvNS_16Flash_fwd_paramsE"),
        "stack_bytes_per_thread": 176,
        "scalar_load_sites": 53,
        "scalar_store_sites": 51,
        "artifact_name": "production_d128_causal",
        "mechanism": (
            "sm80_128_thread_cta_176_byte_stack_51_scalar_store_"
            "53_scalar_load_round"),
    },
}
RAW_FIELDS = (
    "trial", "grid", "rounds", "resident_limit", "active_sms",
    "min_ctas", "max_ctas", "median_block_cycles", "max_sm_span_cycles",
    "checksum",
)
GRIDS = (108, 216)
ROUNDS = (0, 1, 4, 16, 64, 256)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd, capture_output=False, stdout=None, stderr=None):
    return subprocess.run(
        command, cwd=cwd, check=True, text=True,
        capture_output=capture_output, stdout=stdout, stderr=stderr,
        timeout=600)


def _validated_output(path):
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be below {root}")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def _device_rows():
    result = _run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.used,utilization.gpu",
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


def _compute_processes():
    result = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT, capture_output=True)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if row:
            if len(row) != 3:
                raise ValueError("unexpected compute-process row")
            rows.append((row[0], int(row[1]), int(row[2])))
    return rows


def _require_idle_device(index):
    matches = [row for row in _device_rows() if row["index"] == index]
    if len(matches) != 1:
        raise ValueError(f"device {index} does not resolve exactly once")
    device = matches[0]
    if device["name"] != EXPECTED_DEVICE:
        raise ValueError(f"unexpected device: {device['name']}")
    if device["memory_used_mib"] > 8 or device["utilization_percent"] != 0:
        raise RuntimeError(f"device {index} is not idle: {device}")
    processes = [row for row in _compute_processes()
                 if row[0] == device["uuid"]]
    if processes:
        raise RuntimeError(f"device {index} has compute processes: {processes}")
    return device


def _wait_for_idle_device(index):
    last_error = None
    for _ in range(10):
        try:
            return _require_idle_device(index)
        except RuntimeError as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"device {index} did not return idle: {last_error}")


def _local_site_counts(sass):
    return {
        "scalar_load_sites": len(re.findall(r"\sLDL(?:\.|\s)", sass)),
        "scalar_store_sites": len(re.findall(r"\sSTL(?:\.|\s)", sass)),
    }


def _compile(output, source, *, control, specialization):
    name = "register_control" if control else "local_spill"
    binary = output / "build" / name
    command = [
        NVCC, "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo",
        "-Xptxas=-O0,-v",
    ]
    if control:
        command.append("-DREGISTER_CONTROL")
    if specialization == "d128_causal":
        command.append("-DD128_CAUSAL")
    command.extend((str(source), "-o", str(binary)))
    with (output / "build" / f"{name}.compile.stdout").open("w") as stdout, (
            output / "build" / f"{name}.compile.stderr").open("w") as stderr:
        _run(command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    resources = _run(
        [CUOBJDUMP, "--dump-resource-usage", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True).stdout
    sass = _run(
        [CUOBJDUMP, "--dump-sass", str(binary)],
        cwd=REPOSITORY_ROOT, capture_output=True).stdout
    (output / "build" / f"{name}.resources.txt").write_text(resources)
    sass_path = output / "build" / f"{name}.sass"
    sass_path.write_text(sass)
    stack_matches = re.findall(r"STACK:(\d+)", resources)
    if len(stack_matches) != 1:
        raise ValueError(f"{name} resource record changed")
    sites = _local_site_counts(sass)
    contract = PRODUCTION_CONTRACTS[specialization]
    expected = {
        "stack_bytes_per_thread": (
            0 if control else contract["stack_bytes_per_thread"]),
        "scalar_load_sites": (
            0 if control else contract["scalar_load_sites"]),
        "scalar_store_sites": (
            0 if control else contract["scalar_store_sites"]),
    }
    actual = {"stack_bytes_per_thread": int(stack_matches[0]), **sites}
    if actual != expected:
        raise ValueError(f"{name} static contract changed: {actual}")
    return binary, sass_path, actual


def _capture(binary, output, device, trials, name):
    raw = output / f"{name}.csv"
    with raw.open("w") as stdout, (output / f"{name}.stderr").open("w") as stderr:
        _run([str(binary), str(device), str(trials)],
             cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    with raw.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"{name} raw header changed")
        rows = list(reader)
    if len(rows) != trials * len(GRIDS) * len(ROUNDS):
        raise ValueError(f"{name} raw row count changed")
    normalized = []
    for row in rows:
        value = {field: int(row[field]) for field in RAW_FIELDS}
        if value["grid"] not in GRIDS or value["rounds"] not in ROUNDS:
            raise ValueError(f"{name} contains an unknown cell")
        expected_ctas = value["grid"] // EXPECTED_SM_COUNT
        if (value["active_sms"] != EXPECTED_SM_COUNT
                or value["min_ctas"] != expected_ctas
                or value["max_ctas"] != expected_ctas):
            raise ValueError(f"{name} CTA assignment changed")
        if value["resident_limit"] < 2:
            raise ValueError(f"{name} cannot admit two resident CTAs")
        normalized.append(value)
    return raw, normalized


def _median(rows, grid, rounds):
    values = [row["median_block_cycles"] for row in rows
              if row["grid"] == grid and row["rounds"] == rounds]
    if not values:
        raise ValueError("missing measurement cell")
    if (max(values) - min(values)) / median(values) > 0.05:
        raise ValueError("measurement cell is unstable")
    return float(median(values))


def _production_contract(output, binary, specialization):
    if _sha256(binary) != EXPECTED_PRODUCTION_BINARY_SHA256:
        raise ValueError("production FlashAttention binary SHA256 changed")
    extracted = output / "build" / "production_cubins"
    extracted.mkdir()
    _run([CUOBJDUMP, "--extract-elf", "all", str(binary)], cwd=extracted)
    contract = PRODUCTION_CONTRACTS[specialization]
    symbol = contract["symbol"]
    matches = []
    for cubin in extracted.glob("*.sm_80.cubin"):
        symbols = _run(
            [CUOBJDUMP, "--dump-elf-symbols", str(cubin)],
            cwd=REPOSITORY_ROOT, capture_output=True).stdout
        if symbol in symbols:
            matches.append(cubin)
    if len(matches) != 1:
        raise ValueError(
            f"production {specialization} specialization is not unique")
    cubin = matches[0]
    resources = _run(
        [CUOBJDUMP, "--dump-resource-usage", str(cubin)],
        cwd=REPOSITORY_ROOT, capture_output=True).stdout
    resource_lines = resources.splitlines()
    indices = [index for index, line in enumerate(resource_lines)
               if line.strip() == f"Function {symbol}:"]
    if len(indices) != 1:
        raise ValueError("production resource record is not unique")
    resource = "\n".join(resource_lines[indices[0]:indices[0] + 2]) + "\n"
    expected_resource = (
        f"REG:255 STACK:{contract['stack_bytes_per_thread']}")
    if expected_resource not in resource:
        raise ValueError("production stack/register contract changed")
    sass = _run([
        CUOBJDUMP, "--dump-sass", "--function", symbol,
        str(cubin),
    ], cwd=REPOSITORY_ROOT, capture_output=True).stdout
    sites = _local_site_counts(sass)
    expected_sites = {
        "scalar_load_sites": contract["scalar_load_sites"],
        "scalar_store_sites": contract["scalar_store_sites"],
    }
    if sites != expected_sites:
        raise ValueError(f"production local-site contract changed: {sites}")
    artifact_name = contract["artifact_name"]
    sass_path = output / "build" / f"{artifact_name}.sass"
    sass_path.write_text(sass)
    (output / "build" / f"{artifact_name}.resources.txt").write_text(
        resource)
    return {"binary_sha256": _sha256(binary),
            "sass_sha256": _sha256(sass_path),
            "registers_per_thread": 255,
            "stack_bytes_per_thread": contract["stack_bytes_per_thread"],
            **sites}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--production-binary", type=Path, required=True)
    parser.add_argument(
        "--specialization", choices=tuple(PRODUCTION_CONTRACTS),
        default="d64_chunked")
    args = parser.parse_args()
    if args.device < 0 or not 3 <= args.trials <= 11:
        raise ValueError("device must be nonnegative and trials must be in [3, 11]")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", args.owner):
        raise ValueError("invalid owner")
    output = _validated_output(args.output)
    device = _require_idle_device(args.device)
    output.mkdir(parents=True)
    (output / "build").mkdir()
    (output / "source_snapshot").mkdir()
    source = output / "source_snapshot" / SOURCE_PATH.name
    runner = output / "source_snapshot" / RUNNER_PATH.name
    shutil.copy2(SOURCE_PATH, source)
    shutil.copy2(RUNNER_PATH, runner)
    local_binary, local_sass, local_static = _compile(
        output, source, control=False,
        specialization=args.specialization)
    control_binary, control_sass, control_static = _compile(
        output, source, control=True,
        specialization=args.specialization)
    production = _production_contract(
        output, args.production_binary.resolve(), args.specialization)
    local_raw, local_rows = _capture(
        local_binary, output, args.device, args.trials, "local")
    control_raw, control_rows = _capture(
        control_binary, output, args.device, args.trials, "control")
    increments = {}
    for grid in GRIDS:
        increments[str(grid)] = {
            str(rounds): _median(local_rows, grid, rounds)
            - _median(control_rows, grid, rounds)
            for rounds in ROUNDS
        }
    one_round = tuple(increments[str(grid)]["1"] for grid in GRIDS)
    steady = tuple(
        (increments[str(grid)]["256"] - increments[str(grid)]["64"])
        / (256 - 64) for grid in GRIDS)
    if (not math.isclose(one_round[0], one_round[1], abs_tol=1.0)
            or not math.isclose(
                steady[0], steady[1], abs_tol=4.0 / (256 - 64))):
        raise ValueError(
            "one-CTA and two-CTA spill services disagree beyond clock64 "
            "median endpoint resolution")
    if one_round[0] <= 0 or steady[0] <= 0:
        raise ValueError("spill service must be positive")
    commit = _run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout.strip()
    contract = PRODUCTION_CONTRACTS[args.specialization]
    report = {
        "schema": "nvidia_a800_local_spill_round_candidate_v2",
        "hardware_id": "nvidia_a800", "device_index": args.device,
        "device_uuid": device["uuid"], "baseline_commit": commit,
        "owner": args.owner, "specialization": args.specialization,
        "mechanism": contract["mechanism"],
        "production_static_contract": production,
        "probe_static_contract": {
            "source_sha256": _sha256(source),
            "local_binary_sha256": _sha256(local_binary),
            "local_sass_sha256": _sha256(local_sass),
            "control_binary_sha256": _sha256(control_binary),
            "control_sass_sha256": _sha256(control_sass),
            **local_static,
            "control_stack_bytes_per_thread": control_static[
                "stack_bytes_per_thread"],
            "control_local_sites": (
                control_static["scalar_load_sites"]
                + control_static["scalar_store_sites"]),
        },
        "measurement": {
            "timer_domain": "clock64_same_cta_envelope_cycles",
            "trials": args.trials, "grid_blocks": list(GRIDS),
            "rounds": list(ROUNDS), "local_minus_control_cycles": increments,
            "one_round_local_minus_register_control_cycles": one_round[0],
            "steady_local_minus_register_control_cycles_per_round": steady[0],
            "one_cta_two_cta_max_clock64_tick_difference": max(
                abs(one_round[0] - one_round[1]),
                abs(steady[0] - steady[1]) * (256 - 64)),
            "local_raw_sha256": _sha256(local_raw),
            "control_raw_sha256": _sha256(control_raw),
        },
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "A source-matched local stack with the recorded scalar store/load "
            "site geometry adds the measured SM cycles for its single "
            "complete 128-thread CTA spill round."),
        "excluded_conclusions": [
            "production FlashAttention duration", "per-shape latency correction",
            "local spill service for a different stack or SASS site geometry",
        ],
        "provenance": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "pid": os.getpid(), "runner_sha256": _sha256(runner),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    _wait_for_idle_device(args.device)


if __name__ == "__main__":
    main()
