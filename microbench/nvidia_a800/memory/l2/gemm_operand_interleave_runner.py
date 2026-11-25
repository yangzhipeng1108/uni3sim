#!/usr/bin/env python3
"""Capture non-duration L2/DRAM counters for independent operand interleave."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


SOURCE = Path(__file__).with_name("gemm_operand_interleave.cu")
RUNNER = Path(__file__).resolve()
NVCC = "/usr/local/cuda-12.2/bin/nvcc"
NCU = "/usr/local/cuda-12.2/bin/ncu"
CUOBJDUMP = "/usr/local/cuda-12.2/bin/cuobjdump"
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
METRICS = (
    "dram__bytes_read.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_ltcfabric_op_read.sum",
    "smsp__inst_executed_pipe_tensor.sum",
)
CELLS = tuple(
    (m_tiles, 64, include_a, async_copy)
    for m_tiles in (8, 32)
    for include_a in (0, 1)
    for async_copy in (0, 1)
) + tuple(
    (m_tiles, n_tiles, 1, 1)
    for n_tiles in (32, 96)
    for m_tiles in (8, 32)
)
TRIALS = 3


def _run(argv: list[str], *, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        argv, capture_output=True, text=True, env=environment, timeout=600,
    )
    if result.returncode:
        raise RuntimeError(
            f"command {argv[0]} failed with exit {result.returncode}: "
            f"{result.stderr.strip()}")
    return result.stdout


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _device(device: int) -> dict[str, object]:
    output = _run([
        "nvidia-smi", f"--id={device}",
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError("A800 device query has unexpected shape")
    index, name, uuid, memory_mib, utilization = rows[0]
    if int(index) != device or name != DEVICE_NAME:
        raise ValueError("A800 device identity differs")
    if int(memory_mib) > 8 or int(utilization) != 0:
        raise RuntimeError("A800 device is occupied")
    processes = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ])
    if any(line.split(",", 1)[0].strip() == uuid
           for line in processes.splitlines()):
        raise RuntimeError("A800 device has a compute process")
    return {"index": device, "name": name, "uuid": uuid}


def _postflight_device(device: int) -> dict[str, object]:
    for attempt in range(30):
        try:
            return _device(device)
        except RuntimeError:
            if attempt == 29:
                raise
            time.sleep(0.5)
    raise AssertionError("unreachable A800 postflight state")


def _byte_metric(row: dict[str, str], units: dict[str, str], name: str) -> int:
    factors = {"byte": 1, "Kbyte": 1000, "Mbyte": 1000000,
               "Gbyte": 1000000000}
    if units[name] not in factors:
        raise ValueError(f"unexpected NCU byte unit for {name}")
    return round(float(row[name]) * factors[units[name]])


def _counter_row(csv_text: str, m_tiles: int, n_tiles: int,
                 include_a: int, async_copy: int, trial: int) -> dict[str, object]:
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if len(rows) != 2:
        raise ValueError("NCU must report units and exactly one probe kernel")
    units, row = rows
    if "operand_interleave_probe" not in row["Kernel Name"]:
        raise ValueError("NCU selected a different kernel")
    if row["Grid Size"] != f"({n_tiles}, {m_tiles}, 1)" or row["Block Size"] != "(256, 1, 1)":
        raise ValueError("NCU probe launch geometry differs")
    tensor_instructions = int(float(row["smsp__inst_executed_pipe_tensor.sum"]))
    if tensor_instructions != 0:
        raise ValueError("operand probe executed a tensor instruction")
    sector_bytes = 32
    return {
        "m_tiles": m_tiles,
        "n_tiles": n_tiles,
        "k_elements": 2048,
        "include_a": bool(include_a),
        "async_copy": bool(async_copy),
        "trial": trial,
        "grid_ctas": n_tiles * m_tiles,
        "a_request_bytes": include_a * n_tiles * m_tiles * 128 * 2048 * 2,
        "b_request_bytes": m_tiles * n_tiles * 256 * 2048 * 2,
        "dram_read_bytes_rounded": _byte_metric(row, units, METRICS[0]),
        "l2_total_read_bytes": int(float(row[METRICS[1]])) * sector_bytes,
        "l2_tex_read_bytes": int(float(row[METRICS[2]])) * sector_bytes,
        "l2_ltcfabric_read_bytes": int(float(row[METRICS[3]])) * sector_bytes,
        "tensor_instruction_count": tensor_instructions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = _device(args.device)
    executable = output / "gemm_operand_interleave"
    compile_argv = [NVCC, "-O3", "-arch=sm_80", "-lineinfo",
                    str(SOURCE), "-o", str(executable)]
    _run(compile_argv)
    sass = _run([CUOBJDUMP, "-sass", str(executable)])
    if "HMMA" in sass or "LDG.E.128" not in sass or "LDGSTS" not in sass:
        raise ValueError("probe SASS does not preserve the copy-only contract")
    (output / "gemm_operand_interleave.sass").write_text(sass)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    rows = []
    commands = []
    for m_tiles, n_tiles, include_a, async_copy in CELLS:
        for trial in range(TRIALS):
            stem = f"m{m_tiles}_n{n_tiles}_a{include_a}_cp{async_copy}_t{trial}"
            report = output / stem
            argv = [
                NCU, "--force-overwrite", "--target-processes", "all",
                "--kernel-name", "regex:operand_interleave_probe",
                "--launch-count", "1", "--metrics", ",".join(METRICS),
                "--export", str(report), str(executable),
                str(m_tiles), str(n_tiles), str(include_a), str(async_copy),
            ]
            commands.append(argv)
            _run(argv, environment=environment)
            csv_text = _run([
                NCU, "--import", str(report) + ".ncu-rep",
                "--page", "raw", "--csv",
            ])
            (output / f"{stem}.csv").write_text(csv_text)
            rows.append(_counter_row(
                csv_text, m_tiles, n_tiles, include_a, async_copy, trial))
    after = _postflight_device(args.device)
    if before != after:
        raise ValueError("A800 device identity changed during capture")
    medians = []
    for m_tiles, n_tiles, include_a, async_copy in CELLS:
        cell = [row for row in rows
                if row["m_tiles"] == m_tiles
                and row["n_tiles"] == n_tiles
                and row["include_a"] is bool(include_a)
                and row["async_copy"] is bool(async_copy)]
        if len(cell) != TRIALS:
            raise ValueError("operand probe trial cardinality differs")
        dram = [row["dram_read_bytes_rounded"] for row in cell]
        median = statistics.median(dram)
        spread = (max(dram) - min(dram)) / median
        if spread > 0.05:
            raise ValueError("operand probe DRAM counter spread exceeds 5%")
        medians.append({
            "m_tiles": m_tiles, "n_tiles": n_tiles,
            "include_a": bool(include_a),
            "async_copy": bool(async_copy),
            "median_dram_read_bytes_rounded": int(median),
            "dram_relative_spread": spread,
        })
    result = {
        "schema_version": 1,
        "mechanism": "sm80_grouped_cta_operand_interleave_traffic",
        "scope": "A/B copy-only, fixed generic tiles and K rounds; no GEMM, no production duration",
        "device": before,
        "rows": rows,
        "medians": medians,
    }
    (output / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    files = [SOURCE, RUNNER, executable, output / "gemm_operand_interleave.sass",
             output / "measurements.json"]
    for m_tiles, n_tiles, include_a, async_copy in CELLS:
        for trial in range(TRIALS):
            stem = f"m{m_tiles}_n{n_tiles}_a{include_a}_cp{async_copy}_t{trial}"
            files.extend((output / f"{stem}.ncu-rep", output / f"{stem}.csv"))
    record = {
        "schema_version": 1,
        "purpose": "Independent A800 cache interleave mechanism counters",
        "device": before,
        "compile_argv": compile_argv,
        "capture_argv": commands,
        "ncu_version": _run([NCU, "--version"]).strip(),
        "boundary": "Only non-duration counters are published. The copy-only program does not establish a production GEMM rate or latency.",
        "file_identity": {str(path): {"bytes": path.stat().st_size,
                                      "sha256": _digest(path)} for path in files},
    }
    (output / "capture_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"medians": medians, "device": before}))


if __name__ == "__main__":
    main()
