#!/usr/bin/env python3
"""Capture non-duration A800 grouped-MoE B-panel fanout counters."""

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
SOURCE = Path(__file__).with_name("fused_moe_b_panel_fanout.cu")
RUNNER = Path(__file__)
OUTPUT_ROOT = ROOT / "profiler_output" / "microbench" / "nvidia_a800"
VLLM_SOURCE = (
    Path(os.environ["MIDLAYER_A800_VLLM_SOURCE"])
    / "model_executor/layers/fused_moe/fused_moe.py"
)
VLLM_SOURCE_SHA256 = (
    "8be8477a653894194ef5c5095afc287b0409a51d980590a36dd43c8471f6740e")
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
BLOCK_THREADS = 128
L2_SECTOR_BYTES = 32
UNIQUE_B_BYTES = 32 * 1024 * 1024
CASES = tuple({
    "case_id": case_id, "case_name": f"b_panel_fanout_{fanout}",
    "fanout": fanout, "experts": 128, "num_pid_m": 128 * fanout,
    "num_pid_n": 1, "cache_state": "cold_evicted",
} for case_id, fanout in enumerate((1, 2, 4, 8)))
NCU_METRIC_UNITS = {
    "lts__t_sectors_srcunit_tex_op_read.sum": "sector",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum": "sector",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum": "sector",
    "dram__bytes_read.sum": "byte",
}
NCU_METRICS = tuple(NCU_METRIC_UNITS)
SOURCE_CONTRACT_SNIPPETS = (
    '"BLOCK_SIZE_M": 64', '"BLOCK_SIZE_N": 64',
    '"BLOCK_SIZE_K": 32', '"GROUP_SIZE_M": 8',
    "num_pid_in_group = GROUP_SIZE_M * num_pid_n",
    "pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)",
    "pid_n = (pid % num_pid_in_group) // group_size_m",
)


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
    if _sha256(path) != VLLM_SOURCE_SHA256:
        raise ValueError("vLLM fused_moe.py SHA256 changed")
    source = path.read_text(encoding="utf-8")
    missing = tuple(snippet for snippet in SOURCE_CONTRACT_SNIPPETS
                    if snippet not in source)
    if missing:
        raise ValueError(f"vLLM fused-MoE source contract changed: {missing}")


def _execute(command, *, environment=None):
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=stderr)
    return {"command": command, "pid": process.pid,
            "stdout": stdout, "stderr": stderr}


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


def _geometry(case):
    panel_bytes = 2048 * 64 * 2
    grid = case["num_pid_m"] * case["num_pid_n"]
    return {
        "grid": grid, "unique_b_bytes": UNIQUE_B_BYTES,
        "logical_b_read_bytes": grid * panel_bytes,
    }


def _parse_native_stdout(text, case):
    rows = tuple(csv.reader(text.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 13:
        raise ValueError("native probe must emit exactly one thirteen-field row")
    row = rows[0]
    observed = {
        "case_name": row[0], "fanout": _parse_uint(row[1], "fanout"),
        "experts": _parse_uint(row[2], "experts"),
        "num_pid_m": _parse_uint(row[3], "num_pid_m"),
        "grid": _parse_uint(row[4], "grid"),
        "tile_m": _parse_uint(row[5], "tile_m"),
        "tile_n": _parse_uint(row[6], "tile_n"),
        "tile_k": _parse_uint(row[7], "tile_k"),
        "group_size_m": _parse_uint(row[8], "group_size_m"),
        "unique_b_bytes": _parse_uint(row[9], "unique_b_bytes"),
        "logical_b_read_bytes": _parse_uint(row[10], "logical_b_read_bytes"),
        "cache_state": row[11], "checksum": _parse_uint(row[12], "checksum"),
    }
    expected = {
        "case_name": case["case_name"], "fanout": case["fanout"],
        "experts": case["experts"], "num_pid_m": case["num_pid_m"],
        "tile_m": 64, "tile_n": 64, "tile_k": 32, "group_size_m": 8,
        "cache_state": case["cache_state"], **_geometry(case),
    }
    for name, value in expected.items():
        if observed[name] != value:
            raise ValueError(f"native {name} disagrees with the frozen case")
    return observed


def _parse_ncu_csv(text, case):
    lines = text.splitlines()
    header_index = None
    required = {"Kernel Name", "Grid Size", "Block Size", *NCU_METRICS}
    for index, line in enumerate(lines):
        if required <= set(next(csv.reader([line]))):
            header_index = index
            break
    if header_index is None:
        raise ValueError("NCU CSV has no recognized wide metric header")
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None or len(reader.fieldnames) != len(
            set(reader.fieldnames)):
        raise ValueError("NCU CSV header is absent or duplicated")
    rows = tuple(reader)
    unit_rows = tuple(row for row in rows if not row["Kernel Name"])
    data_rows = tuple(row for row in rows if row["Kernel Name"])
    if len(unit_rows) != 1 or len(data_rows) != 1:
        raise ValueError("wide NCU CSV requires one unit and one data row")
    data = data_rows[0]
    if data["Kernel Name"] != "profile_fused_moe_b_panel_fanout":
        raise ValueError("NCU selected the wrong kernel")
    if _parse_dimension(data["Grid Size"], "NCU grid") != (
            _geometry(case)["grid"], 1, 1):
        raise ValueError("NCU grid disagrees with the case")
    if _parse_dimension(data["Block Size"], "NCU block") != (
            BLOCK_THREADS, 1, 1):
        raise ValueError("NCU block disagrees with the source contract")
    metrics = {}
    for metric in NCU_METRICS:
        if unit_rows[0][metric] != NCU_METRIC_UNITS[metric]:
            raise ValueError(f"NCU metric unit changed for {metric!r}")
        metrics[metric] = _parse_uint(data[metric], metric)
    return metrics


def _summarize(case, native, metrics):
    l2_sectors = metrics[NCU_METRICS[0]]
    hit_sectors = metrics[NCU_METRICS[1]]
    miss_sectors = metrics[NCU_METRICS[2]]
    dram_bytes = metrics[NCU_METRICS[3]]
    expected_sectors = native["logical_b_read_bytes"] // L2_SECTOR_BYTES
    minimum_sectors = native["unique_b_bytes"] // L2_SECTOR_BYTES
    if not minimum_sectors <= l2_sectors <= expected_sectors:
        raise ValueError("NCU L2 read sectors fall outside B byte bounds")
    if case["fanout"] == 1 and l2_sectors != expected_sectors:
        raise ValueError("fanout-one L2 reads disagree with source B bytes")
    qualifier_delta = l2_sectors - hit_sectors - miss_sectors
    # GA100's lookup-qualified counters need not sum to the total request
    # counter when concurrent requests to one sector merge.  Keep the signed
    # difference explicit; only reject an individual qualifier above total.
    if hit_sectors > l2_sectors or miss_sectors > l2_sectors:
        raise ValueError("NCU L2 lookup qualifier exceeds total requests")
    if dram_bytes % L2_SECTOR_BYTES:
        raise ValueError("NCU DRAM bytes violate 32-byte sector granularity")
    return {
        "case_id": case["case_id"], "case_name": case["case_name"],
        "tile_m": 64, "tile_n": 64, "tile_k": 32, "group_size_m": 8,
        "fanout_ctas_per_expert_b_panel": case["fanout"],
        "experts": case["experts"], "grid": native["grid"],
        "cache_state": case["cache_state"],
        "unique_b_bytes": native["unique_b_bytes"],
        "logical_b_read_bytes": native["logical_b_read_bytes"],
        "l2_read_bytes": l2_sectors * L2_SECTOR_BYTES,
        "l2_ingress_avoided_bytes": (
            native["logical_b_read_bytes"]
            - l2_sectors * L2_SECTOR_BYTES),
        "l2_hit_read_bytes": hit_sectors * L2_SECTOR_BYTES,
        "l2_miss_read_bytes": miss_sectors * L2_SECTOR_BYTES,
        "l2_lookup_qualifier_delta_bytes": qualifier_delta * L2_SECTOR_BYTES,
        "dram_read_bytes": dram_bytes,
        "l2_hit_fraction": hit_sectors / l2_sectors,
        "l2_read_to_logical_b_fraction": l2_sectors / expected_sectors,
        "dram_to_unique_b_fraction": dram_bytes / native["unique_b_bytes"],
        "dram_to_logical_b_fraction": (
            dram_bytes / native["logical_b_read_bytes"]),
    }


def _device(device_id):
    result = _execute([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,"
        "memory.used,clocks.current.sm,driver_version",
        "--format=csv,noheader,nounits",
    ])
    fields = tuple(field.strip() for field in result["stdout"].strip().split(","))
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
    processes = _execute([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ])["stdout"]
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
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument(
        "--ncu", type=Path, default=Path("/usr/local/cuda-12.2/bin/ncu"))
    parser.add_argument("--vllm-source", type=Path, default=VLLM_SOURCE)
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
    binary = output_dir / "fused_moe_b_panel_fanout"
    compile_command = [
        str(arguments.nvcc.resolve()), "-O3", "-std=c++17", "-arch=sm_80",
        str(SOURCE), "-o", str(binary),
    ]
    compile_execution = _execute(compile_command)
    campaign_before = _wait_idle(arguments.device_id)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    summaries = []
    executions = []
    for case in CASES:
        before = _wait_idle(arguments.device_id)
        native = _execute(
            [str(binary), str(case["case_id"])], environment=environment)
        native_row = _parse_native_stdout(native["stdout"], case)
        csv_path = ncu_dir / f"{case['case_name']}.csv"
        ncu_command = [
            str(arguments.ncu.resolve()), "--kernel-name-base", "function",
            "--kernel-name", "regex:^profile_fused_moe_b_panel_fanout$",
            "--replay-mode", "application", "--cache-control", "none",
            "--clock-control", "none", "--launch-count", "1", "--metrics",
            ",".join(NCU_METRICS), "--page", "raw", "--csv",
            "--print-units", "base", "--log-file", str(csv_path),
            "--force-overwrite", str(binary), str(case["case_id"]),
        ]
        ncu = _execute(ncu_command, environment=environment)
        metrics = _parse_ncu_csv(csv_path.read_text(encoding="utf-8"), case)
        summaries.append(_summarize(case, native_row, metrics))
        executions.append({
            "case_id": case["case_id"], "device_before": before,
            "device_after": _device(arguments.device_id),
            "native_execution": native, "ncu_execution": ncu,
        })
    campaign_after = _wait_idle(arguments.device_id)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    baseline = _execute(["git", "rev-parse", "HEAD"])["stdout"].strip()
    manifest = {
        "schema": "a800_fused_moe_b_panel_fanout_capture_v1",
        "hardware_task": {
            "hardware": "nvidia_a800", "owner": arguments.owner,
            "device": campaign_before, "baseline_commit": baseline,
            "output_dir": str(output_dir),
        },
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256(SOURCE),
        "runner_path": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": _sha256(RUNNER), "binary_sha256": _sha256(binary),
        "vllm_source_path": str(vllm_source),
        "vllm_source_sha256": _sha256(vllm_source),
        "vllm_source_contract_snippets": SOURCE_CONTRACT_SNIPPETS,
        "compile_execution": compile_execution, "executions": executions,
        "invocation_command": sys.argv,
        "campaign_device_before": campaign_before,
        "campaign_device_after": campaign_after,
        "summary_sha256": _sha256(summary_path),
        "ncu_metrics": NCU_METRICS, "ncu_cache_control": "none",
        "ncu_replay_mode": "application",
        "production_or_evaluation_duration_consumed": False,
        "production_counter_consumed": False,
        "production_kernel_reproduced": False,
        "measurement_domain": "B-panel L2 request/hit/miss and DRAM read bytes",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir), "summaries": summaries,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
