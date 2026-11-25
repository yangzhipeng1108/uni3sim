#!/usr/bin/env python3
"""Capture exact H200 grouped-MoE operand-read signatures.

H200 owner copy of the A800 runner
microbench/nvidia_a800/operators/fused_moe/fused_moe_operand_service_runner.py
as it stood at commit f4476b33b, i.e. the version that produced the A800 raw
``profiler_output/microbench/nvidia_a800/fused_moe_operand_820defa_20260830/
summary.json`` cited by ``grouped_fused_moe_operand_read`` (the current A800
runner is fixed to CASE_ID=1 with an active-row sweep and no longer produces
the BM16 row).  Procedure unchanged: both cases (0: BM16/BN32/BK64, 1:
BM64/BN64/BK32), 3 trials with rotated case order, one process per
(trial, case), 5 CUDA-event samples per process, summary = median of the 15
``logical_bandwidth_gb_s`` samples per case.  Hardware/stack coupling only:
  - build ``-arch=sm_90`` with the remote CUDA 12.9 nvcc (A800: sm_80);
  - the current ``.cu`` (commit ab2bec026) takes ``CASE_ID ACTIVE_ROWS
    SAMPLES``; every row is active (ACTIVE_ROWS = BM), which executes the
    same loads as the f4476b33b two-argument kernel;
  - device check ``NVIDIA H200``; output root
    ``profiler_output/microbench/nvidia_h200``;
  - clock gate: each process runs through
    ``microbench/nvidia_h200/compute/clock_gated_capture.run_gated``
    (600 ms clock warm-up spinner, 50-ms nvidia-smi sampling, rejection unless
    every busy/window sample reads >= 1980 MHz, up to 4 attempts).
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("fused_moe_operand_service.cu")
RUNNER = Path(__file__)
OUTPUT_ROOT = ROOT / "profiler_output" / "microbench" / "nvidia_h200"
sys.path.insert(0, str(ROOT / "microbench/nvidia_h200/compute"))
import clock_gated_capture  # noqa: E402

CASES = (
    (0, "operand_bm16_bn32_bk64", 16, 32, 64),
    (1, "operand_bm64_bn64_bk32", 64, 64, 32),
)
TRIALS = 3
SAMPLES = 5


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse(text, *, trial, definition):
    case_id, name, bm, bn, bk = definition
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if len(fields) != 13 or fields[0] != name:
            raise ValueError(f"unexpected output row: {fields!r}")
        row = {
            "trial": trial, "case_id": case_id, "case_name": name,
            "sample": int(fields[1]), "tile_m": int(fields[2]),
            "tile_n": int(fields[3]), "tile_k": int(fields[4]),
            "grid": int(fields[5]), "active_rows": int(fields[6]),
            "activation_bytes": int(fields[7]),
            "weight_bytes": int(fields[8]),
            "logical_a_read_bytes": int(fields[9]),
            "logical_b_read_bytes": int(fields[10]),
            "milliseconds": float(fields[11]),
            "logical_bandwidth_gb_s": float(fields[12]),
        }
        if ((row["tile_m"], row["tile_n"], row["tile_k"]) != (bm, bn, bk)
                or row["active_rows"] != bm
                or not 0 <= row["sample"] < SAMPLES
                or row["milliseconds"] <= 0
                or row["logical_bandwidth_gb_s"] <= 0):
            raise ValueError(f"row violates the frozen case: {row}")
        rows.append(row)
    if len(rows) != SAMPLES:
        raise ValueError("wrong sample count")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int,
                        choices=(0, 3))
    parser.add_argument("--baseline-commit", required=True)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if (output_dir == OUTPUT_ROOT.resolve()
            or OUTPUT_ROOT.resolve() not in output_dir.parents):
        raise ValueError("output directory must be below the H200 root")
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    binary = output_dir / "fused_moe_operand_service"
    build = clock_gated_capture.build(SOURCE, binary)
    warmup = clock_gated_capture.build_warmup(output_dir)
    device_before = clock_gated_capture.device_identity(arguments.device_id)
    rows = []
    executions = []
    for trial in range(TRIALS):
        order = CASES[trial % len(CASES):] + CASES[:trial % len(CASES)]
        for definition in order:
            name = f"t{trial}_c{definition[0]}"
            command = [str(binary), str(definition[0]), str(definition[2]),
                       str(SAMPLES)]
            verdict = clock_gated_capture.run_gated(
                name, [(command, output_dir / f"{name}.csv")], output_dir,
                arguments.device_id, warmup, timeout=120)
            stdout_path = Path(verdict["stdout_paths"][0])
            validation = Path(f"{stdout_path}.stderr").read_text().strip()
            if not validation.startswith(
                    f"validated case={definition[1]} active_rows="):
                raise ValueError(f"unexpected validation: {validation!r}")
            rows.extend(_parse(stdout_path.read_text(), trial=trial,
                               definition=definition))
            executions.append({"trial": trial, "case_id": definition[0],
                               "command": command, "validation": validation,
                               "clock_verdict": verdict})
    raw = output_dir / "raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for case_id, name, bm, bn, bk in CASES:
        values = [row["logical_bandwidth_gb_s"] for row in rows
                  if row["case_id"] == case_id]
        ordered = sorted(values)
        median = statistics.median(values)
        summaries.append({
            "case_id": case_id, "case_name": name, "tile_m": bm,
            "tile_n": bn, "tile_k": bk, "group_size_m": 1 if bm == 16 else 8,
            "logical_tile_bytes": (bm * bk + bk * bn) * 2,
            "sample_count": len(values),
            "median_logical_bandwidth_gb_s": median,
            "minimum_logical_bandwidth_gb_s": ordered[0],
            "maximum_logical_bandwidth_gb_s": ordered[-1],
            "relative_range_pct": (ordered[-1] - ordered[0]) / median * 100,
        })
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    verdicts = [execution["clock_verdict"] for execution in executions]
    manifest = {
        "schema": "h200_fused_moe_operand_service_capture_v1",
        "hardware_id": "nvidia_h200",
        "baseline_commit": arguments.baseline_commit,
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256(SOURCE), "binary_sha256": _sha256(binary),
        "runner_path": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": _sha256(RUNNER),
        "raw_sha256": _sha256(raw), "summary_sha256": _sha256(summary),
        "build": {key: value for key, value in build.items()
                  if key != "ptxas_verbose"},
        "invocation_command": sys.argv,
        "device_before": device_before,
        "executions": executions, "trials": TRIALS,
        "samples_per_trial": SAMPLES,
        "clock": clock_gated_capture.clock_summary(verdicts),
        "production_or_evaluation_duration_consumed": False,
        "timing_domain": "CUDA event around one operand-read-only kernel",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows),
                      "summaries": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
