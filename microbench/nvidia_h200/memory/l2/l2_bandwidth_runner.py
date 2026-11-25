#!/usr/bin/env python3
"""Capture the H200 sustained L2 read envelope with ``l2_bandwidth.cu``.

Runner for the owner copy ``l2_bandwidth.cu`` (the A800 source had no runner
and its configured value came from NCU, which the H200 host does not have).
The probe times every footprint/grid cell itself with CUDA events (median of
nine trials).  The runner compiles for sm_90, refuses a busy device and runs
``PROCESSES`` independent processes under the SM-clock gate, writing
``measurements.csv`` and ``candidate.json``.  No production shape, operator
or duration is consumed.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as gate  # noqa: E402

NVCC = gate.NVCC
SOURCE = Path(__file__).with_name("l2_bandwidth.cu")
RUNNER = Path(__file__).resolve()
PROCESSES = 2
HEADER = "footprint_mib,grid,block,repetitions,median_ms,min_ms,max_ms,logical_read_gbs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    out = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "l2_bandwidth"
    subprocess.run([str(NVCC), "-O3", "-arch=sm_90", "-cudart", "static",
                    "-o", str(binary), str(SOURCE)], check=True)
    before = gate.require_idle(arguments.device)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rows, meta, clocks, rejected = [], {}, [], []
    for process in range(PROCESSES):
        stdout, accepted, rejects = gate.run_clock_gated(
            [str(binary)], device=arguments.device, log_stem=out / f"p{process}")
        clocks.append(dict(accepted, process=process))
        rejected.extend(dict(r, process=process) for r in rejects)
        lines = stdout.splitlines()
        header = lines.index(HEADER)
        for line in lines[:header]:
            key, value = line.split(",", 1)
            meta[key] = value
        for row in csv.DictReader(lines[header:]):
            rows.append(dict(row, process=process))
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = gate.require_idle(arguments.device, settle_seconds=30.0)
    measurements = out / "measurements.csv"
    measurements.write_text("process," + HEADER + "\n" + "".join(
        f"{r['process']}," + ",".join(r[k] for k in HEADER.split(",")) + "\n" for r in rows))
    candidate = {
        "schema": "nvidia_h200_l2_bandwidth_candidate_v1",
        "hardware_id": "nvidia_h200",
        "production_or_evaluation_duration_consumed": False,
        "probe_meta": meta, "processes": PROCESSES,
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "clock_gate": {"boost_mhz": gate.BOOST_MHZ, "accepted": clocks, "rejected": rejected},
        "source_sha256": gate.sha256(SOURCE), "runner_sha256": gate.sha256(RUNNER),
        "binary_sha256": gate.sha256(binary), "measurements_sha256": gate.sha256(measurements),
    }
    (out / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(row["process"], row["footprint_mib"], row["grid"], row["logical_read_gbs"])
    print("accepted", len(clocks), "rejected", len(rejected))


if __name__ == "__main__":
    main()
