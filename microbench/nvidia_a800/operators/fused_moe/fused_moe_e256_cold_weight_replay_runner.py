#!/usr/bin/env python3
"""Measure cold source-ordered B descriptor traffic without MoE timings."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.kernel.dataset_storage import load_kernel_dataset_document

SOURCE = Path(__file__).with_name("fused_moe_e256_cold_weight_replay.cu")
RUNNER = Path(__file__)
DATASET = ROOT / "kernel_dataset/nvidia_a800/fused_moe.json.zst"
OUTPUT_ROOT = ROOT / "profiler_output/microbench/nvidia_a800"
VLLM_SOURCE = (
    Path(os.environ["MIDLAYER_A800_VLLM_SOURCE"])
    / "model_executor/layers/fused_moe/fused_moe.py"
)
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
GROUP_SIZE_M = 8
EXPERTS = 256
K = 7168
PRODUCTION_N = 4096
COUNTERS = (
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "dram__bytes_read.sum",
)


def _run(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(device_id, *, require_idle):
    result = _run([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,"
        "memory.used,clocks.current.sm,clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ])
    fields = tuple(value.strip() for value in result.stdout.split(","))
    if len(fields) != 9:
        raise ValueError("unexpected nvidia-smi device row")
    row = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "gpu_utilization_pct": int(fields[3]),
        "memory_utilization_pct": int(fields[4]), "memory_used_mib": int(fields[5]),
        "sm_clock_mhz": int(fields[6]), "memory_clock_mhz": int(fields[7]),
        "driver_version": fields[8],
    }
    if row["index"] != device_id or row["name"] != DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    applications = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]).stdout
    has_application = any(
        line.strip() and line.split(",", 1)[0].strip() == row["uuid"]
        for line in applications.splitlines())
    if require_idle and (has_application or row["gpu_utilization_pct"] != 0
                         or row["memory_utilization_pct"] != 0
                         or row["memory_used_mib"] > 8):
        raise RuntimeError(f"target GPU is not idle: {row}")
    return row


def _wait_idle(device_id):
    failures = []
    for _ in range(20):
        try:
            return _device(device_id, require_idle=True)
        except RuntimeError as error:
            failures.append(str(error))
            time.sleep(0.25)
    raise RuntimeError(f"target GPU did not become idle: {failures[-3:]}")


def _source_sha256():
    source = VLLM_SOURCE.read_text(encoding="utf-8")
    snippets = (
        "num_pid_in_group = GROUP_SIZE_M * num_pid_n",
        "pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)",
        "pid_n = (pid % num_pid_in_group) // group_size_m",
        "off_experts * stride_be",
    )
    missing = tuple(snippet for snippet in snippets if snippet not in source)
    if missing:
        raise ValueError(f"vLLM grouped-MoE source changed: {missing}")
    return _sha256(VLLM_SOURCE)


def _source_topology(case_id):
    record = load_kernel_dataset_document(DATASET)[case_id]
    attributes = record["op_info"]["Attr"]
    if (attributes["E"], attributes["TOPK"], attributes["H"], attributes["INT"]) != (
            EXPERTS, 8, K, 2048):
        raise ValueError("case does not have the E256/H7168 W1 source contract")
    tiling = attributes["moe_tiling"]["w1"]
    if (tiling["N"], tiling["K"], tiling["BM"], tiling["BN"], tiling["BK"], tiling["GSM"]) != (
            PRODUCTION_N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
        raise ValueError("case does not have the BM64/BN64/BK32/GSM8 source contract")
    counts = tuple(int(value) for value in attributes["tok_per_expert"])
    if len(counts) != EXPERTS or any(value <= 0 for value in counts):
        raise ValueError("case must have all 256 experts populated")
    m_tiles = tuple(math.ceil(value / BLOCK_M) for value in counts)
    active_m_tiles = sum(m_tiles)
    padded_m_tiles = int(tiling["EM"]) // BLOCK_M
    if (int(tiling["EM"]) % BLOCK_M or active_m_tiles >= padded_m_tiles
            or padded_m_tiles % GROUP_SIZE_M != 4):
        raise ValueError("case does not retain the expected inactive GSM8 tail")
    return {
        "case_id": case_id,
        "tokens": int(attributes["TOK"]),
        "m_tile_counts": m_tiles,
        "active_m_tiles": active_m_tiles,
        "padded_m_tiles": padded_m_tiles,
        "route_count": sum(counts),
    }


def _parse_integer(value, label):
    normalized = value.strip().replace(",", "")
    if re.fullmatch(r"[0-9]+", normalized) is None:
        raise ValueError(f"{label} is not an unsigned integer")
    return int(normalized)


def _parse_csv(path, *, expected_grid, expected_logical_bytes):
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines)
         if {"Kernel Name", "Grid Size", "Block Size", *COUNTERS}
         <= set(next(csv.reader([line])))),
        None)
    if header_index is None:
        raise ValueError("NCU did not emit the cold-weight counter schema")
    rows = tuple(csv.DictReader(lines[header_index:]))
    units = tuple(row for row in rows if not row["Kernel Name"])
    values = tuple(row for row in rows if row["Kernel Name"])
    if len(units) != 1 or len(values) != 1:
        raise ValueError("NCU cold-weight result must contain one data row")
    row = values[0]
    if row["Kernel Name"] != "profile_e256_cold_weight_replay":
        raise ValueError("NCU selected an unexpected kernel")
    if row["Grid Size"] != f"({expected_grid}, 1, 1)" or row["Block Size"] != "(128, 1, 1)":
        raise ValueError("NCU launch differs from the source contract")
    if row["launch__occupancy_limit_shared_mem"] != "5":
        raise ValueError("cold B replay must admit exactly five CTAs per SM")
    if row["launch__shared_mem_per_block_dynamic"] != str(31 * 1024):
        raise ValueError("cold B replay dynamic shared-memory contract changed")
    counters = {name: _parse_integer(row[name], name) for name in COUNTERS}
    l2_read_bytes = counters[COUNTERS[0]] * 32
    # The selected kernel also reads one compact expert-id entry per CTA and
    # writes a one-word checksum.  Those bookkeeping accesses are not B
    # descriptors, but share the raw TEX counter.  Publish them explicitly;
    # allow no more than one 32-byte L2 sector/CTA rather than folding them
    # into B traffic.
    non_b_l2_bytes = l2_read_bytes - expected_logical_bytes
    if non_b_l2_bytes < 0 or non_b_l2_bytes > expected_grid * 32:
        raise ValueError("L2 descriptor bytes do not conserve the source B stream")
    if counters[COUNTERS[3]] % 32:
        raise ValueError("DRAM counter violates the L2 sector size")
    return {
        "l2_read_bytes": l2_read_bytes,
        "b_logical_l2_read_bytes": expected_logical_bytes,
        "non_b_l2_read_bytes": non_b_l2_bytes,
        "l2_hit_bytes": counters[COUNTERS[1]] * 32,
        "l2_miss_bytes": counters[COUNTERS[2]] * 32,
        "dram_read_bytes": counters[COUNTERS[3]],
        "dram_to_l2_read_fraction": counters[COUNTERS[3]] / l2_read_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--n", required=True, type=int)
    arguments = parser.parse_args()
    if arguments.device_id not in (2, 6):
        raise ValueError("A800 cold-weight replay is reserved for devices 2 or 6")
    if arguments.n <= 0 or arguments.n % BLOCK_N or arguments.n == PRODUCTION_N:
        raise ValueError("N must be a non-production positive multiple of 64")
    output_dir = arguments.output_dir.resolve()
    if OUTPUT_ROOT.resolve() not in output_dir.parents or output_dir.exists():
        raise ValueError("output directory must be a new child of the A800 profiler root")
    topology = _source_topology(arguments.case_id)
    vllm_source_sha256 = _source_sha256()
    before = _wait_idle(arguments.device_id)
    output_dir.mkdir(parents=True)
    binary = output_dir / "fused_moe_e256_cold_weight_replay"
    compile_command = [
        "/usr/local/cuda-12.2/bin/nvcc", "-O3", "-std=c++17", "-arch=sm_80",
        str(SOURCE), "-o", str(binary),
    ]
    _run(compile_command)
    count_text = ",".join(str(value) for value in topology["m_tile_counts"])
    native_command = [str(binary), str(arguments.n), str(topology["padded_m_tiles"]),
                      count_text]
    ncu_path = output_dir / "ncu.csv"
    ncu_command = [
        "/usr/local/cuda-12.2/bin/ncu", "--kernel-name-base", "function",
        "--kernel-name", "regex:^profile_e256_cold_weight_replay$",
        "--replay-mode", "application", "--cache-control", "none",
        "--clock-control", "none", "--launch-count", "1", "--metrics", ",".join(COUNTERS),
        "--page", "raw", "--csv", "--print-units", "base", "--log-file", str(ncu_path),
        "--force-overwrite", *native_command,
    ]
    started_at_utc = datetime.now(timezone.utc).isoformat()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    ncu = _run(ncu_command, environment=environment)
    expected_grid = topology["padded_m_tiles"] * (arguments.n // BLOCK_N)
    expected_logical_bytes = (topology["active_m_tiles"] * (arguments.n // BLOCK_N)
                              * K * BLOCK_N * 2)
    counters = _parse_csv(ncu_path, expected_grid=expected_grid,
                          expected_logical_bytes=expected_logical_bytes)
    after = _device(arguments.device_id, require_idle=False)
    payload = {
        "schema": "a800_fused_moe_e256_cold_b_descriptor_counter_v1",
        "owner": "research/kernel-accuracy-fusedmoe",
        "baseline_commit": _run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_or_evaluation_duration_consumed": False,
        "production_kernel_reproduced": False,
        "measurement_domain": "cold B descriptor L2/DRAM traffic only",
        "forbidden_conclusions": ["production duration", "per-case residual", "A/output traffic"],
        "source": {"path": str(SOURCE), "sha256": _sha256(SOURCE),
                   "vllm_path": str(VLLM_SOURCE), "vllm_sha256": vllm_source_sha256},
        "device_before": before, "device_after": after,
        "compile_command": compile_command, "ncu_command": ncu_command,
        "ncu_stdout": ncu.stdout, "ncu_stderr": ncu.stderr,
        "topology": topology, "n": arguments.n, "counters": counters,
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
