#!/usr/bin/env python3
"""Capture A800 grouped-MoE descriptor L2/DRAM byte residency."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("fused_moe_residency_replay.cu")
RUNNER = Path(__file__)
OUTPUT_ROOT = ROOT / "profiler_output" / "microbench" / "nvidia_a800"
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
VLLM_SOURCE = (
    Path(os.environ["MIDLAYER_A800_VLLM_SOURCE"])
    / "model_executor/layers/fused_moe/fused_moe.py"
)
BLOCK_THREADS = 128
L2_BYTES = 40 * 1024 * 1024
L2_SECTOR_BYTES = 32
CASES = (
    {
        "case_id": 0, "case_name": "source_gsm8_resident_32mib",
        "num_pid_m": 8, "num_pid_n": 15, "cache_state": "source_warmed",
    },
    {
        "case_id": 1, "case_name": "source_gsm8_over_l2_48mib",
        "num_pid_m": 12, "num_pid_n": 15, "cache_state": "source_warmed",
    },
    {
        "case_id": 2, "case_name": "source_gsm8_over_l2_64mib",
        "num_pid_m": 16, "num_pid_n": 15, "cache_state": "source_warmed",
    },
    {
        "case_id": 3, "case_name": "source_gsm8_cold_32mib_control",
        "num_pid_m": 8, "num_pid_n": 15, "cache_state": "cold_evicted",
    },
)
NCU_METRIC_UNITS = {
    "lts__t_sectors_srcunit_tex_op_read.sum": "sector",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum": "sector",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum": "sector",
    "dram__bytes_read.sum": "byte",
}
NCU_METRICS = tuple(NCU_METRIC_UNITS)
SOURCE_ORDER_SNIPPETS = (
    "num_pid_in_group = GROUP_SIZE_M * num_pid_n",
    "pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)",
    "pid_n = (pid % num_pid_in_group) // group_size_m",
)


def _run(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_vllm_source(path):
    source = path.read_text(encoding="utf-8")
    missing = tuple(snippet for snippet in SOURCE_ORDER_SNIPPETS
                    if snippet not in source)
    if missing:
        raise ValueError(f"vLLM grouped descriptor order changed: {missing}")


def _parse_uint(text, label):
    normalized = text.strip().replace(",", "")
    if not re.fullmatch(r"[0-9]+", normalized):
        raise ValueError(f"{label} must be an unsigned integer")
    return int(normalized)


def _parse_dimension(text, label):
    match = re.fullmatch(
        r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)",
        text.strip())
    if match is None:
        raise ValueError(f"{label} has invalid CUDA dimension syntax")
    return tuple(int(match.group(index)) for index in range(1, 4))


def _parse_ncu_csv(text, case):
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        fields = set(next(csv.reader([line])))
        tall_fields = {
            "Kernel Name", "Grid Size", "Block Size", "Metric Name",
            "Metric Unit", "Metric Value",
        }
        wide_fields = {
            "Kernel Name", "Grid Size", "Block Size", *NCU_METRICS,
        }
        if tall_fields <= fields or wide_fields <= fields:
            header_index = index
            break
    if header_index is None:
        raise ValueError("NCU CSV has no recognized metric header")
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None or len(reader.fieldnames) != len(
            set(reader.fieldnames)):
        raise ValueError("NCU CSV header is absent or duplicated")
    expected_grid = case["num_pid_m"] * case["num_pid_n"]
    rows = tuple(reader)

    def validate_launch(row):
        if row["Kernel Name"] != "profile_fused_moe_descriptor_replay":
            raise ValueError("NCU selected the wrong kernel")
        if _parse_dimension(row["Grid Size"], "NCU grid") != (
                expected_grid, 1, 1):
            raise ValueError("NCU grid disagrees with the case")
        if _parse_dimension(row["Block Size"], "NCU block") != (
                BLOCK_THREADS, 1, 1):
            raise ValueError("NCU block disagrees with the source contract")

    fields = set(reader.fieldnames)
    tall_fields = {
        "Kernel Name", "Grid Size", "Block Size", "Metric Name",
        "Metric Unit", "Metric Value",
    }
    wide_fields = {
        "Kernel Name", "Grid Size", "Block Size", *NCU_METRICS,
    }
    metrics = {}
    if wide_fields <= fields:
        unit_rows = tuple(row for row in rows if not row["Kernel Name"])
        data_rows = tuple(row for row in rows if row["Kernel Name"])
        if len(unit_rows) != 1 or len(data_rows) != 1:
            raise ValueError("wide NCU CSV requires one unit and one data row")
        validate_launch(data_rows[0])
        for metric in NCU_METRICS:
            if unit_rows[0][metric] != NCU_METRIC_UNITS[metric]:
                raise ValueError(f"NCU metric unit changed for {metric!r}")
            metrics[metric] = _parse_uint(data_rows[0][metric], metric)
    elif tall_fields <= fields:
        for row in rows:
            validate_launch(row)
            metric = row["Metric Name"]
            if metric not in NCU_METRIC_UNITS or metric in metrics:
                raise ValueError(f"NCU metric identity is invalid: {metric!r}")
            if row["Metric Unit"] != NCU_METRIC_UNITS[metric]:
                raise ValueError(f"NCU metric unit changed for {metric!r}")
            metrics[metric] = _parse_uint(row["Metric Value"], metric)
    else:
        raise ValueError("NCU CSV lacks the required tall or wide schema")
    if set(metrics) != set(NCU_METRICS):
        raise ValueError("NCU metric schema is incomplete")
    return metrics


def _expected_geometry(case):
    activation_bytes = case["num_pid_m"] * 64 * 2048 * 2
    weight_bytes = case["num_pid_m"] * 2048 * 960 * 2
    grid = case["num_pid_m"] * case["num_pid_n"]
    logical_descriptor_bytes = grid * 2048 * (64 + 64) * 2
    return {
        "grid": grid,
        "activation_bytes": activation_bytes,
        "weight_bytes": weight_bytes,
        "unique_operand_bytes": activation_bytes + weight_bytes,
        "logical_descriptor_bytes": logical_descriptor_bytes,
    }


def _parse_native_stdout(text, case):
    rows = tuple(csv.reader(text.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 10:
        raise ValueError("native probe must emit exactly one ten-field row")
    row = rows[0]
    geometry = _expected_geometry(case)
    observed = {
        "case_name": row[0], "num_pid_m": _parse_uint(row[1], "num_pid_m"),
        "num_pid_n": _parse_uint(row[2], "num_pid_n"),
        "grid": _parse_uint(row[3], "grid"),
        "activation_bytes": _parse_uint(row[4], "activation_bytes"),
        "weight_bytes": _parse_uint(row[5], "weight_bytes"),
        "unique_operand_bytes": _parse_uint(row[6], "unique_operand_bytes"),
        "logical_descriptor_bytes": _parse_uint(
            row[7], "logical_descriptor_bytes"),
        "cache_state": row[8], "checksum": _parse_uint(row[9], "checksum"),
    }
    expected = {
        "case_name": case["case_name"], "num_pid_m": case["num_pid_m"],
        "num_pid_n": case["num_pid_n"], "cache_state": case["cache_state"],
        **geometry,
    }
    for name, value in expected.items():
        if observed[name] != value:
            raise ValueError(f"native {name} disagrees with the frozen case")
    return observed


def _summarize(case, native, metrics):
    l2_sectors = metrics[NCU_METRICS[0]]
    hit_sectors = metrics[NCU_METRICS[1]]
    miss_sectors = metrics[NCU_METRICS[2]]
    dram_bytes = metrics[NCU_METRICS[3]]
    expected_sectors = native["logical_descriptor_bytes"] // L2_SECTOR_BYTES
    if l2_sectors != expected_sectors:
        raise ValueError("NCU L2 read sectors disagree with descriptor bytes")
    classified_sectors = hit_sectors + miss_sectors
    unclassified_sectors = l2_sectors - classified_sectors
    # Nsight's lookup qualifiers omit a small number of sectors at request
    # boundaries.  Admit at most 512 bytes per CTA and publish the remainder
    # explicitly instead of folding it into either hits or misses.
    if abs(unclassified_sectors) > native["grid"] * 16:
        raise ValueError("NCU L2 lookup qualifier delta is too large")
    if dram_bytes % L2_SECTOR_BYTES:
        raise ValueError("NCU DRAM bytes violate 32-byte sector granularity")
    return {
        "case_id": case["case_id"], "case_name": case["case_name"],
        "tile_m": 64, "tile_n": 64, "tile_k": 32, "group_size_m": 8,
        "num_pid_m": case["num_pid_m"], "num_pid_n": case["num_pid_n"],
        "grid": native["grid"], "cache_state": case["cache_state"],
        "l2_capacity_bytes": L2_BYTES,
        "unique_operand_bytes": native["unique_operand_bytes"],
        "unique_operand_exceeds_l2": (
            native["unique_operand_bytes"] > L2_BYTES),
        "logical_descriptor_read_bytes": native["logical_descriptor_bytes"],
        "l2_read_bytes": l2_sectors * L2_SECTOR_BYTES,
        "l2_hit_read_bytes": hit_sectors * L2_SECTOR_BYTES,
        "l2_miss_read_bytes": miss_sectors * L2_SECTOR_BYTES,
        "l2_lookup_qualifier_delta_bytes": (
            unclassified_sectors * L2_SECTOR_BYTES),
        "dram_read_bytes": dram_bytes,
        "l2_hit_fraction": hit_sectors / l2_sectors,
        "dram_to_logical_read_fraction": (
            dram_bytes / native["logical_descriptor_bytes"]),
        "dram_to_unique_operand_fraction": (
            dram_bytes / native["unique_operand_bytes"]),
    }


def _device(device_id):
    query = subprocess.run([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,"
        "memory.used,clocks.current.sm,driver_version",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True)
    fields = tuple(field.strip() for field in query.stdout.strip().split(","))
    if len(fields) != 8:
        raise ValueError("unexpected nvidia-smi device row")
    row = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "gpu_utilization_pct": int(fields[3]),
        "memory_utilization_pct": int(fields[4]),
        "memory_used_mib": int(fields[5]), "sm_clock_mhz": int(fields[6]),
        "driver_version": fields[7],
    }
    if row["index"] != device_id or row["name"] != DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    processes = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True).stdout
    if any(
            line.strip() and line.split(",", 1)[0].strip() == row["uuid"]
            for line in processes.splitlines()):
        raise RuntimeError("target GPU has a compute process")
    return row


def _wait_idle(device_id):
    observations = []
    for _ in range(10):
        row = _device(device_id)
        observations.append(row)
        if (row["gpu_utilization_pct"] == 0
                and row["memory_utilization_pct"] == 0
                and row["memory_used_mib"] <= 8):
            return row
        time.sleep(0.5)
    raise RuntimeError(f"GPU did not become idle: {observations}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument(
        "--ncu", type=Path, default=Path("/usr/local/cuda-12.2/bin/ncu"))
    parser.add_argument(
        "--vllm-source", type=Path, default=VLLM_SOURCE)
    arguments = parser.parse_args()
    vllm_source = arguments.vllm_source.resolve()
    _validate_vllm_source(vllm_source)
    output_dir = arguments.output_dir.resolve()
    if (output_dir == OUTPUT_ROOT.resolve()
            or OUTPUT_ROOT.resolve() not in output_dir.parents):
        raise ValueError("output directory must be below the A800 root")
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    ncu_dir = output_dir / "ncu"
    ncu_dir.mkdir()
    binary = output_dir / "fused_moe_residency_replay"
    compile_command = [
        str(arguments.nvcc.resolve()), "-O3", "-std=c++17", "-arch=sm_80",
        str(SOURCE), "-o", str(binary),
    ]
    _run(compile_command)
    campaign_before = _wait_idle(arguments.device_id)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    summaries = []
    executions = []
    for case in CASES:
        before = _wait_idle(arguments.device_id)
        native_command = [str(binary), str(case["case_id"])]
        native_result = _run(native_command, environment=environment)
        native = _parse_native_stdout(native_result.stdout, case)
        csv_path = ncu_dir / f"{case['case_name']}.csv"
        ncu_command = [
            str(arguments.ncu.resolve()), "--kernel-name-base", "function",
            "--kernel-name", "regex:^profile_fused_moe_descriptor_replay$",
            "--replay-mode", "application", "--cache-control", "none",
            "--clock-control", "none", "--launch-count", "1", "--metrics",
            ",".join(NCU_METRICS), "--page", "raw", "--csv",
            "--print-units", "base", "--log-file", str(csv_path),
            "--force-overwrite", str(binary), str(case["case_id"]),
        ]
        ncu_result = _run(ncu_command, environment=environment)
        metrics = _parse_ncu_csv(csv_path.read_text(encoding="utf-8"), case)
        summaries.append(_summarize(case, native, metrics))
        after = _device(arguments.device_id)
        executions.append({
            "case_id": case["case_id"], "native_command": native_command,
            "ncu_command": ncu_command, "native_stdout": native_result.stdout,
            "native_stderr": native_result.stderr,
            "ncu_stdout": ncu_result.stdout, "ncu_stderr": ncu_result.stderr,
            "device_before": before, "device_after": after,
        })
    campaign_after = _wait_idle(arguments.device_id)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    manifest = {
        "schema": "a800_fused_moe_residency_replay_capture_v1",
        "baseline_commit": _run(
            ["git", "rev-parse", "HEAD"]).stdout.strip(),
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256(SOURCE),
        "vllm_source_path": str(vllm_source),
        "vllm_source_sha256": _sha256(vllm_source),
        "vllm_source_order_snippets": SOURCE_ORDER_SNIPPETS,
        "runner_path": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": _sha256(RUNNER), "binary_sha256": _sha256(binary),
        "summary_sha256": _sha256(summary_path),
        "compile_command": compile_command, "invocation_command": sys.argv,
        "campaign_device_before": campaign_before,
        "campaign_device_after": campaign_after, "executions": executions,
        "ncu_metrics": NCU_METRICS, "ncu_cache_control": "none",
        "ncu_replay_mode": "application",
        "production_or_evaluation_duration_consumed": False,
        "production_kernel_reproduced": False,
        "measurement_domain": "L2 request/hit/miss and DRAM read bytes only",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir), "summaries": summaries,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
