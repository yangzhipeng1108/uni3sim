#!/usr/bin/env python3
"""Capture the independent two-resident-CTA finite LDGSTS service."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


HARDWARE_ID = "nvidia_a800"
OWNER = "research/kernel-accuracy-fa worker"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("fa2_resident_ldgsts_finite.cu")
REPORT_PATH = Path(__file__).with_name(
    "fa2_resident_ldgsts_finite_report.py")
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / HARDWARE_ID)
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
REPEATS = (1, 2, 4, 8, 16, 32, 64)
SAMPLES = 7
NCU_METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "lts__t_bytes.sum",
    "lts__t_sector_hit_rate.pct",
    "sm__cycles_elapsed.avg.per_second",
    "sm__sass_inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd=REPOSITORY_ROOT, env=None):
    return subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True,
        check=True)


def _parse_uint(text, name):
    normalized = text.strip().replace(",", "")
    if re.fullmatch(r"[0-9]+", normalized) is None:
        raise ValueError(f"{name} must be an unsigned integer")
    return int(normalized)


def _device_row(device_id, *, require_idle):
    result = _run([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.used,utilization.gpu,"
        "clocks.current.sm,clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ])
    rows = tuple(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 9:
        raise ValueError("nvidia-smi returned an unexpected device row")
    values = tuple(value.strip() for value in rows[0])
    row = {
        "index": _parse_uint(values[0], "GPU index"),
        "name": values[1],
        "uuid": values[2],
        "pci_bus_id": values[3],
        "memory_used_mib": _parse_uint(values[4], "GPU memory used"),
        "utilization_gpu_pct": _parse_uint(values[5], "GPU utilization"),
        "sm_clock_mhz": _parse_uint(values[6], "GPU SM clock"),
        "memory_clock_mhz": _parse_uint(values[7], "GPU memory clock"),
        "driver_version": values[8],
    }
    if row["index"] != device_id or row["name"] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    applications = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]).stdout
    if any(line.split(",", 1)[0].strip() == row["uuid"]
           for line in applications.splitlines() if line.strip()):
        raise RuntimeError("target GPU already has a compute process")
    if require_idle and (
            row["memory_used_mib"] != 0 or row["utilization_gpu_pct"] != 0):
        raise RuntimeError(f"target GPU is not idle: {row}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", type=int, choices=(1, 5), default=1)
    parser.add_argument(
        "--nvcc", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument(
        "--cuobjdump", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/cuobjdump"))
    parser.add_argument(
        "--ncu", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/ncu"))
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if (output_dir == OUTPUT_ROOT.resolve()
            or OUTPUT_ROOT.resolve() not in output_dir.parents):
        raise ValueError("output directory must be below the A800 microbench root")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    for tool in (arguments.nvcc, arguments.cuobjdump, arguments.ncu):
        if not tool.is_file():
            raise FileNotFoundError(f"capture tool does not exist: {tool}")

    device = _device_row(arguments.device_id, require_idle=True)
    output_dir.mkdir(parents=True)
    binary_path = output_dir / "fa2_resident_ldgsts_finite"
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_80",
        "--ptxas-options=-v", str(SOURCE_PATH), "-o", str(binary_path),
    ]
    compiled = _run(compile_command)
    ptxas_path = output_dir / "ptxas.txt"
    ptxas_path.write_text(compiled.stderr, encoding="utf-8")
    sass_path = output_dir / "probe.sass"
    sass = _run([str(arguments.cuobjdump), "--dump-sass", str(binary_path)])
    sass_path.write_text(sass.stdout, encoding="utf-8")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    commands = []
    for repeats in REPEATS:
        csv_path = output_dir / f"final_r{repeats}.csv"
        report_path = output_dir / f"final_r{repeats}.ncu-rep"
        command = [
            str(arguments.ncu), "--kernel-name-base", "function",
            "--kernel-name", "regex:fa2_resident_ldgsts_finite",
            "--cache-control", "none", "--clock-control", "none",
            "--launch-skip", "10", "--launch-count", str(SAMPLES),
            "--metrics", ",".join(NCU_METRICS), "--page", "raw", "--csv",
            "--print-units", "base", "--log-file", str(csv_path),
            "--export", str(report_path), "--force-overwrite",
            str(binary_path), str(repeats), str(SAMPLES),
        ]
        _run(command, env=environment)
        commands.append(command)

    tmux_session = None
    if "TMUX" in os.environ:
        tmux_session = os.environ["TMUX"]
    manifest = {
        "schema": "a800_fa2_resident_ldgsts_finite_capture_v1",
        "hardware_id": HARDWARE_ID,
        "owner": OWNER,
        "device": device,
        "baseline_commit": _run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "command": [str(Path(__file__).resolve()), *os.sys.argv[1:]],
        "pid": os.getpid(),
        "tmux": tmux_session,
        "compile_command": compile_command,
        "capture_commands": commands,
        "source_sha256": _sha256(SOURCE_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "reporter_sha256": _sha256(REPORT_PATH),
        "binary_sha256": _sha256(binary_path),
        "ptxas_sha256": _sha256(ptxas_path),
        "sass_sha256": _sha256(sass_path),
        "repeats": list(REPEATS),
        "samples_per_cell": SAMPLES,
        "threads_per_cta": 128,
        "grid_ctas": 216,
        "dynamic_shared_bytes": 77824,
        "panel_bytes_per_cta": 65536,
        "production_or_evaluation_duration_consumed": False,
        "cleanup": "complete",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
